import argparse


def str2bool(value):
    """Parse boolean command-line values safely."""
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "1"):
        return True
    if value in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def configs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--max_epoches', type=int, default=501)
    parser.add_argument('--save_dir', type=str, default='model_data')
    parser.add_argument('--generate_image', type=str2bool, default=False)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument("--lr-beta1", type=float, default=0.95, help="learning rate beta 1")
    parser.add_argument("--lr-beta2", type=float, default=0.99, help="learning rate beta 2")
    parser.add_argument("--l2-norm", type=float, default=0, help="l2 norm weight decay")
    parser.add_argument("--mixed_precision", type=str, default='fp16', help="mixed precision training")
    return parser
