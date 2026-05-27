import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from harp_film.dataset import (
    TIME_KEYS,
    TECPredictDataset,
    build_dataloader,
    build_valid_indices,
    fit_aux_scaler,
    fit_tec_scaler,
    infer_step,
    load_timestamps,
    read_aux_keys,
    save_scalers,
)
from harp_film.model import CNNEncoder, Configs
from harp_film.train_eval import (
    EarlyStopping,
    compute_metrics,
    evaluate_loss,
    predict_original_scale,
    save_results_to_hdf5,
    train_one_epoch,
)
from harp_film.utils import count_trainable_parameters, load_yaml, save_json, set_random_seed, setup_logging


def main():
    parser = argparse.ArgumentParser(description="Train HARP-FiLM.")
    parser.add_argument("--config", type=str, default="configs/harp_film.yaml", help="Path to YAML config file.")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger("train_harp_film")
    cfg = load_yaml(args.config)
    set_random_seed(int(cfg.get("seed", 42)))

    data_cfg = cfg["data"]
    win_cfg = cfg["window"]
    loader_cfg = cfg["loader"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ts = load_timestamps(data_cfg["train_hdf5_path"])
    val_ts = load_timestamps(data_cfg["val_hdf5_path"])
    test_ts = load_timestamps(data_cfg["test_hdf5_path"])

    train_indices = build_valid_indices(
        train_ts,
        win_cfg["input_timestamps"],
        win_cfg["output_timestamps"],
        win_cfg["stride_train"],
        infer_step(train_ts.to_numpy(dtype="datetime64[s]")),
    )
    val_indices = build_valid_indices(
        val_ts,
        win_cfg["input_timestamps"],
        win_cfg["output_timestamps"],
        win_cfg["stride_val"],
        infer_step(val_ts.to_numpy(dtype="datetime64[s]")),
    )
    test_indices = build_valid_indices(
        test_ts,
        win_cfg["input_timestamps"],
        win_cfg["output_timestamps"],
        win_cfg["stride_test"],
        infer_step(test_ts.to_numpy(dtype="datetime64[s]")),
    )
    logger.info("Window counts: train=%d, val=%d, test=%d", len(train_indices), len(val_indices), len(test_indices))

    selected_aux_keys = data_cfg.get("selected_aux_keys")
    aux_keys = read_aux_keys(data_cfg["train_hdf5_path"], selected_aux_keys)
    feature_names = aux_keys + TIME_KEYS
    logger.info("Auxiliary feature order: %s", feature_names)

    tec_scaler = fit_tec_scaler(
        data_cfg["train_hdf5_path"],
        train_indices,
        win_cfg["input_timestamps"],
        data_cfg["num_latitudes"],
        data_cfg["num_longitudes"],
    )
    aux_scaler = fit_aux_scaler(data_cfg["train_hdf5_path"], aux_keys, train_indices, win_cfg["input_timestamps"])
    save_scalers(output_dir, tec_scaler, aux_scaler, aux_keys)
    save_json({"feature_names": feature_names}, output_dir / "feature_names.json")

    common_ds = dict(
        input_timestamps=win_cfg["input_timestamps"],
        output_timestamps=win_cfg["output_timestamps"],
        tec_scaler=tec_scaler,
        fixed_aux_keys=aux_keys,
        aux_scaler=aux_scaler,
        num_latitudes=data_cfg["num_latitudes"],
        num_longitudes=data_cfg["num_longitudes"],
    )
    train_dataset = TECPredictDataset(data_cfg["train_hdf5_path"], indices=train_indices, **common_ds)
    val_dataset = TECPredictDataset(data_cfg["val_hdf5_path"], indices=val_indices, **common_ds)
    test_dataset = TECPredictDataset(data_cfg["test_hdf5_path"], indices=test_indices, **common_ds)

    pin_memory = bool(loader_cfg.get("pin_memory", torch.cuda.is_available())) and torch.cuda.is_available()
    num_workers = int(loader_cfg.get("num_workers", 0)) if pin_memory else 0
    batch_size = int(loader_cfg["batch_size"])
    train_loader = build_dataloader(train_dataset, batch_size, True, num_workers, pin_memory)
    val_loader = build_dataloader(val_dataset, batch_size, False, num_workers, pin_memory)
    test_loader = build_dataloader(test_dataset, batch_size, False, num_workers, pin_memory)

    configs = Configs(
        seq_len=win_cfg["input_timestamps"],
        label_len=win_cfg["input_timestamps"],
        pred_len=win_cfg["output_timestamps"],
        d_model=model_cfg["d_model"],
        d_ff=model_cfg["d_ff"],
        n_heads=model_cfg["n_heads"],
        e_layers=model_cfg["e_layers"],
        dropout=model_cfg["dropout"],
        activation=model_cfg["activation"],
        input_feature_dim=len(feature_names),
        k_recent=model_cfg["k_recent"],
        gain_limit=model_cfg["gain_limit"],
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CNNEncoder(
        configs,
        cnn_channels=model_cfg["cnn_channels"],
        num_latitudes=data_cfg["num_latitudes"],
        num_longitudes=data_cfg["num_longitudes"],
        input_feature_dim=configs.input_feature_dim,
        use_learnable_temporal=model_cfg.get("use_learnable_temporal", True),
    ).to(device)
    if torch.cuda.device_count() > 1:
        logger.info("Using DataParallel on %d GPUs.", torch.cuda.device_count())
        model = torch.nn.DataParallel(model)
    logger.info("Trainable parameters: %.2f M", count_trainable_parameters(model) / 1e6)

    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=train_cfg["scheduler_factor"],
        patience=train_cfg["scheduler_patience"],
        min_lr=train_cfg["min_learning_rate"],
    )
    early_stopping = EarlyStopping(patience=train_cfg["early_stopping_patience"], verbose=True)
    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    history = []
    best_val_loss = float("inf")
    checkpoint_path = output_dir / train_cfg["checkpoint_name"]
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, train_cfg["grad_clip_norm"])
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        lr = optimizer.param_groups[0]["lr"]
        logger.info("Epoch %03d | train_loss=%.6f | val_loss=%.6f | lr=%g", epoch, train_loss, val_loss, lr)
        writer.add_scalar("Train/Loss", train_loss, epoch)
        writer.add_scalar("Validation/Loss", val_loss, epoch)
        writer.add_scalar("Train/Learning_Rate", lr, epoch)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr})
        scheduler.step(val_loss)
        early_stopping(val_loss, model)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(state, checkpoint_path)
            logger.info("Saved best checkpoint to %s", checkpoint_path)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered.")
            break
    writer.close()
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

    # Load best checkpoint and evaluate on test split.
    state_dict = torch.load(checkpoint_path, map_location=device)
    if isinstance(model, torch.nn.DataParallel):
        state_dict = {("module." + k if not k.startswith("module.") else k): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    predictions, targets = predict_original_scale(model, test_loader, device, tec_scaler)
    metrics = compute_metrics(predictions, targets)
    logger.info("Test metrics: %s", metrics)
    save_json(metrics, output_dir / "test_metrics.json")

    save_results_to_hdf5(
        predictions=predictions,
        targets=targets,
        test_indices=test_indices,
        input_timestamps=win_cfg["input_timestamps"],
        timestamps=test_dataset.timestamps,
        lat_grid=test_dataset.lat_grid,
        lon_grid=test_dataset.lon_grid,
        output_hdf5_path=output_dir / train_cfg["prediction_hdf5_name"],
    )


if __name__ == "__main__":
    main()
