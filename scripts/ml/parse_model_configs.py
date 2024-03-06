#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-28 10:33:56 (ywatanabe)"

import argparse

import mngs

CONFIG = mngs.gen.load_configs(show=False)


def parse_model_configs():
    parser = argparse.ArgumentParser(
        description="Train the EEGPT model with given hyperparameters."
    )

    # Create groups for better organization
    general_group = parser.add_argument_group("General Settings")
    conv_group = parser.add_argument_group("Convolution Layer Settings")
    vq_group = parser.add_argument_group("Vector Quantization Settings")
    transformer_group = parser.add_argument_group("Transformer Settings")

    # Transformer settings
    transformer_group.add_argument(
        "--tf_input_dim",
        type=int,
        default=84,
        help="Input dimension for transformer (default: %(default)s)",
    )
    transformer_group.add_argument(
        "--tf_hidden_dim",
        type=int,
        default=64,
        help="Hidden dimension for transformer (default: %(default)s)",
    )
    transformer_group.add_argument(
        "--tf_dim_feedforward",
        type=int,
        default=2048,
        help="Dimension of feedforward network in transformer (default: %(default)s)",
    )
    transformer_group.add_argument(
        "--tf_n_heads",
        type=int,
        default=2,
        help="Number of attention heads in transformer (default: %(default)s)",
    )
    transformer_group.add_argument(
        "--tf_n_enc_layers",
        type=int,
        default=2,
        help="Number of encoder layers in transformer (default: %(default)s)",
    )
    transformer_group.add_argument(
        "--tf_n_dec_layers",
        type=int,
        default=2,
        help="Number of decoder layers in transformer (default: %(default)s)",
    )
    transformer_group.add_argument(
        "--tf_dropout_ratio",
        type=float,
        default=0.1,
        help="Dropout ratio in transformer (default: %(default)s)",
    )

    # Parse the arguments
    args_dict = vars(parser.parse_args())

    # Some adjustments
    print("\n" + "-" * 40)
    for k, v in args_dict.items():
        print(
            k,
        )
        print(v)
        print()
    # print(args_dict)
    print("\n" + "-" * 40)

    return args_dict


def parse_model_configs_v1(pretrain_stage=None):
    parser = argparse.ArgumentParser(
        description="Train the EEGPT model with given hyperparameters."
    )

    # Create groups for better organization
    general_group = parser.add_argument_group("General Settings")
    conv_group = parser.add_argument_group("Convolution Layer Settings")
    vq_group = parser.add_argument_group("Vector Quantization Settings")
    transformer_group = parser.add_argument_group("Transformer Settings")

    # General settings
    # general_group.add_argument(
    #     "--learning_rate",
    #     type=float,
    #     default=1e-3,
    #     help="Initial learning rate (default: %(default)s)",
    # )
    # general_group.add_argument(
    #     "--batch_size",
    #     type=int,
    #     default=32,
    #     help="Batch size for training (default: %(default)s)",
    # )
    # general_group.add_argument(
    #     "--optimizer",
    #     default="SGD",
    #     help="Optimizer type (default: %(default)s)",
    # )
    # general_group.add_argument(
    #     "--n_epochs",
    #     type=int,
    #     default=100,
    #     help="Number of training epochs (default: %(default)s)",
    # )

    # Convolution Layer settings
    conv_group.add_argument(
        "--conv_n_blocks",
        type=int,
        default=3,
        help="Number of convolution blocks (default: %(default)s)",
    )
    conv_group.add_argument(
        "--conv_split_size_pts",
        type=int,
        default=32,
        help="Size of splits for convolution blocks in points (default: %(default)s)",
    )

    # Vector Quantization settings
    vq_group.add_argument(
        "--vq_num_embeddings",
        type=int,
        default=4,
        help="Number of VQ embeddings (default: %(default)s)",
    )
    vq_group.add_argument(
        "--vq_embedding_dim",
        type=int,
        default=16,
        help="Dimension of VQ embeddings (default: %(default)s)",
    )
    vq_group.add_argument(
        "--vq_initializer",
        type=str,
        default="uniform",
        help="Initializer for VQ embeddings (default: %(default)s)",
    )
    vq_group.add_argument(
        "--vq_weight_norm",
        type=bool,
        default=False,
        help="Apply weight normalization in VQ (default: %(default)s)",
    )
    vq_group.add_argument(
        "--vq_softmax_type",
        type=str,
        default="gumbel_softmax_hard",
        help="Type of softmax used in VQ (default: %(default)s)",
    )
    vq_group.add_argument(
        "--vq_commitment_coef",
        type=float,
        default=1.0,
        help="Commitment coefficient for VQ (default: %(default)s)",
    )
    vq_group.add_argument(
        "--vq_entropy_coef",
        type=float,
        default=1.0,
        help="Entropy coefficient for VQ (default: %(default)s)",
    )
    vq_group.add_argument(
        "--vq_diversity_coef",
        type=float,
        default=1.0,
        help="Diversity coefficient for VQ (default: %(default)s)",
    )
    vq_group.add_argument(
        "--vq_prediction_coef",
        type=float,
        default=1.0,
        help="Prediction coefficient for VQ (default: %(default)s)",
    )

    # Transformer settings
    transformer_group.add_argument(
        "--tf_hidden_dim",
        type=int,
        default=64,
        help="Hidden dimension for transformer (default: %(default)s)",
    )
    transformer_group.add_argument(
        "--tf_dim_feedforward",
        type=int,
        default=2048,
        help="Dimension of feedforward network in transformer (default: %(default)s)",
    )
    transformer_group.add_argument(
        "--tf_n_heads",
        type=int,
        default=2,
        help="Number of attention heads in transformer (default: %(default)s)",
    )
    transformer_group.add_argument(
        "--tf_n_enc_layers",
        type=int,
        default=2,
        help="Number of encoder layers in transformer (default: %(default)s)",
    )
    transformer_group.add_argument(
        "--tf_n_dec_layers",
        type=int,
        default=2,
        help="Number of decoder layers in transformer (default: %(default)s)",
    )
    transformer_group.add_argument(
        "--tf_dropout_ratio",
        type=float,
        default=0.1,
        help="Dropout ratio in transformer (default: %(default)s)",
    )

    # Parse the arguments
    args_dict = vars(parser.parse_args())

    # Some adjustments
    args_dict["tf_input_dim"] = args_dict["vq_embedding_dim"]
    args_dict["pretrain_stage"] = pretrain_stage
    args_dict = mask_loss_coefs(args_dict)

    print("\n" + "-" * 40)
    for k, v in args_dict.items():
        print(
            k,
        )
        print(v)
        print()
    # print(args_dict)
    print("\n" + "-" * 40)

    return args_dict


def mask_loss_coefs(args_dict):
    # if args_dict["pretrain_stage"] is None:
    #     return args_dict

    if args_dict["pretrain_stage"] == "conv_and_vq":
        args_dict["vq_prediction_coef"] = 0
        return args_dict

    if args_dict["pretrain_stage"] == "transformer":
        args_dict["vq_commitment_coef"] = 0
        args_dict["vq_entropy_coef"] = 0
        args_dict["vq_diversity_coef"] = 0
        return args_dict

    else:
        return args_dict


if __name__ == "__main__":
    args = parse_model_configs()
    print(f"\nargs:\n{args}")

    args = parse_model_configs(pretrain_stage="conv_and_vq")
    print(f"\nargs:\n{args}")

    args = parse_model_configs(pretrain_stage="transformer")
    print(f"\nargs:\n{args}")

    """
    ./scripts/ml/parse_model_configs.py -h | tee ./docs/model_config_arguments.txt
    """
