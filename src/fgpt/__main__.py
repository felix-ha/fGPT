import sys
import argparse
import fgpt
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

    parser_training = sub_parsers.add_parser('train', help='start or continue training')
    parser_training.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs",
    )

    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    if args.subcommand == "data-pipeline":
        print("data")
    elif args.subcommand == "train":
        print(f"bar: {bar(args.epochs)}")


if __name__ == "__main__":
    main()
