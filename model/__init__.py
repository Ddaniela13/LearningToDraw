import importlib

from dsketch.experiments.shared.utils import list_class_names
from .agents import *
from .game import *


def get_model(name):
    # load a model class by name
    module = importlib.import_module(__name__)
    return getattr(module, name)


def model_choices(clz):
    return list(filter(lambda x: not x.startswith('_'), list_class_names(clz, __name__)))
