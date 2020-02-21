import argparse
import os
from pathlib import Path


def parse_hyperparameters(hm):
    """Convert list of ['--name', 'value', ...] to { 'name': value}, where 'value' is converted to the nearest data type.

    Conversion follows the principle: "if it looks like a duck and quacks like a duck, then it must be a duck".
    """
    d = {}

    it = iter(hm)
    try:
        while True:
            key = next(it)[2:]
            value = next(it)
            d[key] = value
    except StopIteration:
        pass

    # Infer data types.
    dd = {k: infer_dtype(v) for k, v in d.items()}
    return dd


def infer_dtype(s):
    """Auto-cast string values to nearest matching datatype.

    Conversion follows the principle: "if it looks like a duck and quacks like a duck, then it must be a duck".

    Note that python 3.6 implements PEP-515 which allows '_' as thousand separators. Hence, on Python 3.6,
    '1_000' is a valid number and will be converted accordingly.
    """
    if s == "None":
        return None
    if s == "True":
        return True
    if s == "False":
        return False

    try:
        i = float(s)
        if ("." in s) or ("e" in s.lower()):
            return i
        else:
            return int(s)
    except:  # noqa:E722
        pass

    return s


def assert_train_args(args):
    ...


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Minimum args according to SageMaker protocol.
    parser.add_argument("--model_dir", type=Path, default=os.environ.get("SM_MODEL_DIR", "model"))
    parser.add_argument("--train", type=Path, default=os.environ.get("SM_CHANNEL_TRAIN", "train"))
    parser.add_argument("--test", type=Path, default=os.environ.get("SM_CHANNEL_TEST", "test"))

    args, train_args = parser.parse_known_args()

    ...
