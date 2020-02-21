import argparse
import os
import sys
from pathlib import Path

#### Make tqdm more quiet ####
# This stanza must be before import run_ner or any module that uses tqdm.
# https://github.com/tqdm/tqdm/issues/619#issuecomment-425234504
import tqdm
old_tqdm = tqdm.tqdm
old_trange = tqdm.trange
def nop_tqdm(*a, **k):
    k['ncols'] = 0
    k['file'] = sys.stdout
    return old_tqdm(*a, **k)

def nop_trange(*a, **k):
    k['ncols'] = 0
    k['file'] = sys.stdout
    return old_trange(*a, **k)
tqdm.tqdm=nop_tqdm
tqdm.trange=nop_trange
#### End of quiet tqdm ####

import run_ner


def assert_train_args(args):
    locked_args = {"--do_train", "--do-eval", "--evaluate_during_train", "--data_dir", "--output_dir", "--label"}
    violation = {s for s in args if s[0] == "-"} & locked_args
    if len(violation) > 0:
        raise ValueError(f"Error: overriden args {violation} in locked args {locked_args}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Minimum args according to SageMaker protocol.
    parser.add_argument("--model_dir", default=os.environ.get("SM_MODEL_DIR", "model"))
    parser.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAIN", "train"))
    parser.add_argument("--dev", default=os.environ.get("SM_CHANNEL_DEV", ""))
    parser.add_argument("--label", default=os.environ.get("SM_CHANNEL_LABEL", None))
    # Rename "-h" flag for run_ner.py
    parser.add_argument("--train-help", action="store_true")

    args, train_args = parser.parse_known_args()

    # Patch CLI args to pass down to run_ner.py.
    if args.train_help:
        # Note the argparse -h behavior: run_ner.main() will show help, then exit!
        sys.argv = ["run_ner.py", "-h"]
    else:
        assert_train_args(train_args)
        cmd_opts = ["run_ner.py", "--do_train"]

        # run_ner.py expects `data_dir` to contain these files:
        # - train.txt: mandatory for --do_train
        # - dev.txt: mandatory for either --do_eval or --evaluate_during_train
        # - test.txt: mandatory for --do_predict (which is unsupported by this script)
        data_dir = ("--data_dir", args.train)

        # (Train + evaluate) requested
        if args.dev != "":
            cmd_opts += ["--do_eval"] #, "--evaluate_during_train"]
            try:
                os.link(
                    os.path.join(args.dev, "dev.txt"), os.path.join(args.train, "dev.txt")
                )  # Copy (by hard link) the dev data to match with what run_ner.py expects.
            except FileExistsError:
                print(os.path.join(args.dev, "dev.txt"), ' already exists; skip hard linking')

        label = ["--label", os.path.join(args.label, "label.txt")] if args.label else []
        model_dir = ("--output_dir", args.model_dir)

        # Recommended additional args: --model_type, --model_name_or_path, --num_train_epochs
        sys.argv = [*cmd_opts, *data_dir, *model_dir, *label, *train_args]

    print(sys.argv)
    run_ner.main()
