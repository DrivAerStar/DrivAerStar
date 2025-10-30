import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Process config')

    parser.add_argument('--config-path', type=str, help='Path to the config file')
    parser.add_argument('--load-ckpt-path', type=str, default=None, help='ckpt path to load')
    parser.add_argument('--use-tb', action='store_true', default=False,)
    parser.add_argument('--use-wandb', action='store_true', default=False)
    parser.add_argument('--wandb-log-key', type=str, default=None)
    args = parser.parse_args()
    return args
