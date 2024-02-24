#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-24 11:04:37 (ywatanabe)"

import argparse

import mngs

CONFIG = mngs.gen.load_configs(show=False)


def parse_model_configs():
    parser = argparse.ArgumentParser(
        description="Train the EEGPT model with given hyperparameters."
    )

    # General hyperparameters
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--optimizer", default="SGD")
    # parser.add_argument("--window_size_sec", type=int, default=None)
    parser.add_argument("--n_epochs", type=int, default=100)

    # Convolution Layers
    parser.add_argument("--conv_n_blocks", type=int, default=3)
    parser.add_argument("--conv_split_size_pts", type=int, default=32)

    # VQ (Vector Quantization) hyperparameters
    parser.add_argument("--vq_num_embeddings", type=int, default=4)
    parser.add_argument("--vq_embedding_dim", type=int, default=16)
    parser.add_argument("--vq_initializer", type=str, default="uniform")
    parser.add_argument("--vq_weight_norm", type=bool, default=False)
    parser.add_argument(
        "--vq_softmax_type", type=str, default="gumbel_softmax_hard"
    )
    parser.add_argument("--vq_commitment_coef", type=float, default=1.0)
    parser.add_argument("--vq_entropy_coef", type=float, default=0.3)
    parser.add_argument("--vq_diversity_coef", type=float, default=1.0)
    parser.add_argument("--vq_prediction_coef", type=float, default=0.0)

    # Transformer hyperparameters
    parser.add_argument("--tf_input_dim", type=int, default=16)
    parser.add_argument("--tf_hidden_dim", type=int, default=64)
    parser.add_argument("--tf_dim_feedforward", type=int, default=2048)
    parser.add_argument("--tf_n_heads", type=int, default=2)
    parser.add_argument("--tf_n_enc_layers", type=int, default=2)
    parser.add_argument("--tf_n_dec_layers", type=int, default=2)
    parser.add_argument("--tf_dropout_ratio", type=float, default=0.1)

    # Parse the arguments
    args = parser.parse_args()
    args_dict = vars(args)

    if not args_dict.get("window_size_pts"):
        CONFIG = mngs.gen.load_configs(show=False)
        args_dict["window_size_pts"] = int(
            CONFIG["WINDOW_SIZE_SEC"] * CONFIG["TGT_FS"]
        )
    return args_dict


if __name__ == "__main__":
    # Parse the command-line arguments
    args = parse_model_configs()

    # Now you can use args to access the command-line arguments
    print(f"\nargs:\n{args}")

    # Continue with the rest of your main function logic
