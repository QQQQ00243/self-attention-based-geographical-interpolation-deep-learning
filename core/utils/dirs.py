import os


def make_dirs(*dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)
    # no return
