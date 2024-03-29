import argparse
import importlib
import os

from .base_task import BaseTask


# register dataclass
TASK_REGISTRY = {}
TASK_CLASS_NAMES = set()


def setup_task_from_args(args):
    return TASK_REGISTRY[args.task].setup(args)


def setup_task_from_file(file_path):
    import json
    with open(file_path, "r") as f:
        args = json.load(f)
        f.close()
    return TASK_REGISTRY[args["task"]].setup(args)


def register_task(name):
    """
    New tasks can be added to unicore with the
    :func:`~core.task.register_task` function decorator.

    For example::

        @register_task('classification')
        class ClassificationTask(UnicoreTask):
            (...)

    .. note::

        All Tasks must implement the :class:`~core.task.BaseTask`
        interface.

    Args:
        name (str): the name of the task
    """

    def register_task_cls(cls):
        if name in TASK_REGISTRY:
            raise ValueError("Cannot register duplicate task ({})".format(name))
        if not issubclass(cls, BaseTask):
            raise ValueError(
                "Task ({}: {}) must extend BaseTask".format(name, cls.__name__)
            )
        if cls.__name__ in TASK_CLASS_NAMES:
            raise ValueError(
                "Cannot register task with duplicate class name ({})".format(
                    cls.__name__
                )
            )
        TASK_REGISTRY[name] = cls
        TASK_CLASS_NAMES.add(cls.__name__)
        return cls

    return register_task_cls


# automatically import any Python files in the tasks/ directory
tasks_dir = os.path.dirname(__file__)
for file in os.listdir(tasks_dir):
    path = os.path.join(tasks_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        task_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module("core.tasks." + task_name)

        '''
        # expose `task_parser` for sphinx
        if task_name in TASK_REGISTRY:
            parser = argparse.ArgumentParser(add_help=False)
            group_task = parser.add_argument_group("Task name")
            # fmt: off
            group_task.add_argument('--task', metavar=task_name,
                                    help='Enable this task with: ``--task=' + task_name + '``')
            # fmt: on
            group_args = parser.add_argument_group("Additional command-line arguments")
            TASK_REGISTRY[task_name].add_args(group_args)
            globals()[task_name + "_parser"] = parser
        '''

