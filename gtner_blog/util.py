import xmlrpc.client

import s3fs

################################################################################
# Data splits
################################################################################
# Default
_fs = s3fs.S3FileSystem(anon=False)


def split(iob_fname, train_fname, test_fname, fs=_fs):
    """Split an .iob located in S3, into train:test = 3:1 splits, then save to two S3 objects."""
    with fs.open(iob_fname, "r") as f_iob:
        with fs.open(train_fname, "w") as f_train:
            with fs.open(test_fname, "w") as f_test:
                for i, sentence in enumerate(sentences(f_iob)):
                    f_split = f_train if (i % 3) < 2 else f_test
                    f_split.write("\n".join(sentence))
                    f_split.write("\n\n")  # 1x \n for the last token, and 1x \n as a sentence separator


def sentences(f):
    """This function assumes .iob file does not contain document marker -DOCSTART-."""
    sentence = []
    for line in f:
        line = line.rstrip()
        if line == "":
            yield sentence
            sentence = []
        else:
            sentence.append(line)
    yield sentence


################################################################################
# Package versions
################################################################################
def get_latest_version(name: str) -> str:
    pypi = xmlrpc.client.ServerProxy("https://pypi.python.org/pypi")
    return pypi.package_releases("transformers")[0]
