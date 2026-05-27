import torch
from harp_film.model import CNNEncoder, Configs


def main():
    batch_size = 2
    seq_len = 24
    pred_len = 24
    H, W = 71, 73
    feature_dim = 9  # five selected drivers + four temporal encodings

    configs = Configs(seq_len=seq_len, pred_len=pred_len, input_feature_dim=feature_dim)
    model = CNNEncoder(configs, cnn_channels=12, num_latitudes=H, num_longitudes=W, input_feature_dim=feature_dim)

    x_dec = torch.randn(batch_size, seq_len, 5, H, W)
    x_mark_dec = torch.randn(batch_size, seq_len, feature_dim)
    y = model(x_dec, x_mark_dec)
    print("Output shape:", tuple(y.shape))  # expected: (B, 24, 71, 73)

    y_diag, diagnostics = model(x_dec, x_mark_dec, return_diagnostics=True)
    print("Diagnostics keys:", diagnostics.keys())


if __name__ == "__main__":
    main()
