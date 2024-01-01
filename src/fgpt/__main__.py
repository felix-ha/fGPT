import sys
import argparse
import fgpt
from fgpt.data import data_pipeline
from fgpt.foo import bar


def main():
    print(f"Welcome to fgpt, version {fgpt.__version__}")
    parser = argparse.ArgumentParser(prog='fgpt', description='A tool to train a lm')
    
    sub_parsers = parser.add_subparsers(help='sub-command help', dest="subcommand")

    parser_data = sub_parsers.add_parser('data-pipeline', help='run data pipeline')
    parser_data.add_argument(
        "--full",
        default=False,
        action="store_true",
        help="Use full TinyStores dataset instead of the small one.",
    )
    parser_data.add_argument(
        "--data_path",
        type=str,
        default="datapipeline",
        help="Output path for the data pipeline",
    )
    parser_data.add_argument(
        "--n_vocab",
        type=int,
        default=10_000,
        help="Number of words in the vocabulary",
    )
    parser_data.add_argument(
        "--n_texts_per_partition",
        type=int,
        default=100_000,
        help="Number of texts per partition",
    )
    parser_data.add_argument(
        "--partition_size",
        type=str,
        default="100MB",
        help="Size of each partition",
    )

    parser_training = sub_parsers.add_parser('train', help='start or continue training')
    parser_training.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs",
    )

    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    if args.subcommand == "data-pipeline":
        data_pipeline(args.data_path, args.full, args.n_vocab, args.n_texts_per_partition, args.partition_size)
    elif args.subcommand == "train":
        print(f"bar: {bar(args.epochs)}")


if __name__ == "__main__":
    main()
