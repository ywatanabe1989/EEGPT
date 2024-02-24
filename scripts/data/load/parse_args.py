import argparse


class CustomArgs:
    def __init__(self):
        self.gen = argparse.Namespace()
        self.cnn = argparse.Namespace()
        self.vq = argparse.Namespace()
        self.tf = argparse.Namespace()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Example with custom grouped arguments."
    )

    # General parameters
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--window_size_sec", type=float, default=1.0)
    parser.add_argument("--n_epochs", type=int, default=10)

    # CNN parameters
    parser.add_argument("--cnn_dropout_ratio", type=float, default=0.1)

    # VQ parameters
    parser.add_argument("--vq_num_embeddings", type=int, default=64)
    parser.add_argument("--vq_embedding_dim", type=int, default=128)
    parser.add_argument(
        "--vq_softmax_type", type=str, default="gumbel_softmax_hard"
    )
    parser.add_argument("--vq_commit_coef", type=float, default=1.0)
    parser.add_argument("--vq_entropy_coef", type=float, default=0.3)
    parser.add_argument("--vq_pred_coef", type=float, default=0.0)

    # Transformer parameters
    parser.add_argument("--tf_n_heads", type=int, default=8)
    parser.add_argument("--tf_hidden_dim", type=int, default=512)
    parser.add_argument("--tf_n_enc_layers", type=int, default=6)
    parser.add_argument("--tf_n_dec_layers", type=int, default=6)
    parser.add_argument("--tf_dropout_ratio", type=float, default=0.1)

    args = parser.parse_args()

    # Assign parsed arguments to CustomArgs
    custom_args = CustomArgs()
    custom_args.gen = argparse.Namespace(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        optimizer=args.optimizer,
        window_size_sec=args.window_size_sec,
        n_epochs=args.n_epochs,
    )
    custom_args.cnn = argparse.Namespace(
        cnn_dropout_ratio=args.cnn_dropout_ratio
    )
    custom_args.vq = argparse.Namespace(
        vq_num_embeddings=args.vq_num_embeddings,
        vq_embedding_dim=args.vq_embedding_dim,
        vq_softmax_type=args.vq_softmax_type,
        vq_commit_coef=args.vq_commit_coef,
        vq_entropy_coef=args.vq_entropy_coef,
        vq_pred_coef=args.vq_pred_coef,
    )
    custom_args.tf = argparse.Namespace(
        tf_n_heads=args.tf_n_heads,
        tf_hidden_dim=args.tf_hidden_dim,
        tf_n_enc_layers=args.tf_n_enc_layers,
        tf_n_dec_layers=args.tf_n_dec_layers,
        tf_dropout_ratio=args.tf_dropout_ratio,
    )

    return custom_args


if __name__ == "__main__":
    custom_args = parse_args()

    # Access the parameters
    print("General Parameters:")
    print(custom_args.gen.learning_rate)
    print(custom_args.gen.batch_size)
    print(custom_args.gen.optimizer)
    print(custom_args.gen.window_size_sec)
    print(custom_args.gen.n_epochs)

    print("\nCNN Parameters:")
    print(custom_args.cnn.cnn_dropout_ratio)

    print("\nVQ Parameters:")
    print(custom_args.vq.vq_num_embeddings)
    print(custom_args.vq.vq_embedding_dim)
    print(custom_args.vq.vq_softmax_type)
    print(custom_args.vq.vq_commit_coef)
    print(custom_args.vq.vq_entropy_coef)
    print(custom_args.vq.vq_pred_coef)

    print("\nTransformer Parameters:")
    print(custom_args.tf.tf_n_heads)
    print(custom_args.tf.tf_hidden_dim)
    print(custom_args.tf.tf_n_enc_layers)
    print(custom_args.tf.tf_n_dec_layers)
    print(custom_args.tf.tf_dropout_ratio)
