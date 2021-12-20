#!/usr/bin/env python
# coding=utf-8
"""
Configuration objects
"""

import importlib
import inspect
import re
import sys
from collections import namedtuple
from numbers import Number
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

from PIL import Image
from .utils import DATA_DIR, SAVE_DIR

__all__ = [
    "BaseConfig",
    "init_transform",
    "init_dataset",
    "init_dataloader",
]

TYPES = [
    "transform",
    "dataset",
    "dataloader",
    "network",
    "loss",
    "lr_scheduler",
    "optimizer",
    "metrics",
]

# BICUBIC = InterpolationMode.BICUBIC
BICUBIC = Image.BICUBIC


class BaseConfig:
    """Base class for all configurations"""
    def __init__(self, **options):
        self._opts: dict = {
            "other": {},
            "global": {},
            "train": {},
            "test": {},
            "val": {},
        }
        self._default_opts: dict = {
            "other": {},
            "global": self.default_global_opts,
            "train": self.default_train_opts,
            "test": self.default_test_opts,
            "val": self.default_val_opts,
        }
        self._stage = options.pop("stage", "other")
        self.opts = options
        self.state: dict = {}
        self.reset()

    @staticmethod
    def is_valid_stage(stage):
        """Verify that the stage is 'other | global | train | test | val'"""
        if stage not in ["other", "global", "train", "test", "val"]:
            raise ValueError("stage not 'other | global | train | test | val'")
        return stage

    def get_opts(self, stage: str = ""):
        """Retrieve the options."""
        if not stage:
            stage = self.is_valid_stage(self.stage)
        else:
            stage = self.is_valid_stage(stage)

        return self._opts[stage]

    def set_opts(self, options: dict, stage: str = ""):
        """Set the options."""
        if not stage:
            stage = self.is_valid_stage(self.stage)
        else:
            stage = self.is_valid_stage(stage)

        self._opts[stage] = options

    opts = property(get_opts, set_opts)

    def get_default_opts(self, stage: str = ""):
        """Retrieve the options."""
        if not stage:
            stage = self.is_valid_stage(self.stage)
        else:
            stage = self.is_valid_stage(stage)

        return self._default_opts[stage]

    default_opts = property(get_default_opts)

    @property
    def default_train_opts(self) -> dict:
        """Default train options"""
        return {}

    @property
    def default_test_opts(self) -> dict:
        """Default test options"""
        return {}

    @property
    def default_val_opts(self) -> dict:
        """Default validation options"""
        return {}

    @property
    def default_global_opts(self) -> dict:
        """Default global options"""
        return {
            "seed": 12345,
            "target_devices": [0],
            "save_dir": SAVE_DIR,
            "data_dir": DATA_DIR,
            "trial_id": "A01-E01-S0001",
            "aim": "hello world",
            "aim_id": "A01",
            "experiment": "pilot",
            "experiment_id": "E01",
            "setup": "baseline",
            "setup_id": "S0001",
            "timestamp": "1970-01-01T00:00:00Z",
        }

    @property
    def stage(self) -> str:
        """
        Returns the stage, i.e., train, test, or val.

        Return
        ------
        _ : str
            Stage name. Either train | test | val
        """
        return self._stage

    @stage.setter
    def stage(self, stage: str) -> None:
        """
        Set the stage. Updates the value of self.stage and reset the state.

        Parameters
        ----------
        value : str
            Stage name. One of train | test | val | global | other.
        """
        self._stage = self.is_valid_stage(stage)
        self.reset()

    @property
    def type(self) -> str:
        """Object type"""
        obj_type = self.get("type")
        if obj_type:
            return obj_type
        return ""

    def set(self, **options):
        """Set options"""
        for key, value in options.items():
            self.state[key] = value
        return self

    def reset(self):
        """Reset options"""
        # Clean state
        self.state = {}
        # Set global options
        self.update(**self.get_default_opts("global"))
        # Update unset options with default values given the stage
        self.update(**self.default_opts)
        # Set user-specified options
        self.set(**self.opts)
        return self

    def get(self, option: str) -> Any:
        """
        Get option. Unknown options return None by default.

        Parameters
        ----------
        option : str
            Option name.

        Returns
        ----------
        _ : Any
            Option current value or None.
        """
        return self.state.get(option, None)

    @property
    def save_dir(self) -> Path:
        """Returns the save directory"""
        trial_id = self.get("trial_dir") if self.get("trial_dir") else ""
        return self.get("save_dir") / trial_id

    @property
    def data_dir(self) -> Path:
        """Return the Path to the dataset dir"""
        dataset = self.get("dataset_name") if self.get("dataset_name") else ""
        return self.get("data_dir") / dataset

    def filter(self, *options):
        """Keep only the options provided"""
        to_rm_opts = [opt for opt in self.state if opt not in options]
        self.remove(*to_rm_opts)
        return self

    def remove(self, *options):
        """Remove options"""
        for option in options:
            if option in self.state:
                del self.state[option]
        return self

    def update(self, **options):
        """Update options"""
        self.state.update(options)
        return self

    def __str__(self):
        string = ""
        for key, val in self.state.items():
            if not isinstance(val, (Number, Sequence, Path, type(None))):
                val = retrieve_name(val)
            string += f"--{key} {val} "
        return string

    def clone(self):
        """
        Clone configuration

        Returns
        -------
        cfg : BaseConfig
        """
        stage = self.stage
        cfg = BaseConfig(stage=stage)
        cfg._opts = self._opts.copy()
        cfg._default_opts = self._default_opts.copy()
        cfg.state = self.state.copy()

        return cfg

    def join(self, other):
        """
        Clone configuration

        Returns
        -------
        cfg : BaseConfig
        """
        if self.type != other.type:
            raise TypeError("Trying to join two Config of different types")
        if self.stage == other.stage:
            print("Warning: joinning two Config with the same stage")
        self.set_opts(other.opts, other.stage)


class TransformConfig(BaseConfig):
    """Transform Configuration object."""
    @property
    def default_train_opts(self) -> dict:
        return {
            "out_ch": 1,
            "load_size": (256, 256),
            "out_size": (128, 128),
            "max_size": (128, 128),
            "crop_size": (128, 128),
            "crop_pos": None,
            "trim_size": (128, 128),
            "patch_size": (128, 128),
            "patch_stride": 0,
            "shortside_size": 128,
            "zoom_factor": None,
            "flip": True,
            "power_base": 4,
            "interp": BICUBIC,
            "dataset_name": "horse2zebra",
        }

    @property
    def default_test_opts(self) -> dict:
        return {
            "out_ch": 1,
            "load_size": (256, 256),
            "out_size": (128, 128),
            "max_size": (128, 128),
            "crop_size": (128, 128),
            "crop_pos": (0, 0),
            "trim_size": (128, 128),
            "patch_size": (128, 128),
            "patch_stride": 0,
            "shortside_size": 128,
            "zoom_factor": (1, 1),
            "power_base": 4,
            "interp": BICUBIC,
            "dataset_name": "horse2zebra",
        }

    @property
    def default_val_opts(self) -> dict:
        return {
            "out_ch": 1,
            "load_size": (256, 256),
            "out_size": (128, 128),
            "max_size": (128, 128),
            "crop_size": (128, 128),
            "crop_pos": (0, 0),
            "trim_size": (128, 128),
            "patch_size": (128, 128),
            "patch_stride": 0,
            "shortside_size": 128,
            "zoom_factor": (1, 1),
            "power_base": 4,
            "interp": BICUBIC,
            "dataset_name": "horse2zebra",
        }


class DatasetConfig(BaseConfig):
    """Dataset Configuration Option"""
    @property
    def default_train_opts(self) -> dict:
        return {
            "max_dataset_size": sys.maxsize,
            "max_class_size": sys.maxsize,
            "return_paths": False,
        }

    @property
    def default_test_opts(self) -> dict:
        return {
            "max_dataset_size": sys.maxsize,
            "max_class_size": sys.maxsize,
            "return_paths": False,
        }

    @property
    def default_val_opts(self) -> dict:
        return {
            "max_dataset_size": sys.maxsize,
            "max_class_size": sys.maxsize,
            "return_paths": False,
        }


class DataloaderConfig(BaseConfig):
    """Dataloader Configuration Option"""
    @property
    def default_train_opts(self) -> dict:
        return {
            "batch_size": 8,
            "shuffle": True,
            "num_workers": 4,
            "drop_last": True,
            "pin_memory": False,
            "n_splits": 5,
            "kfold_shuffle": True,
            "random_state": 0,
        }

    @property
    def default_test_opts(self) -> dict:
        return {
            "batch_size": 8,
            "shuffle": False,
            "num_workers": 4,
            "drop_last": False,
            "pin_memory": False,
            "n_splits": 5,
            "kfold_shuffle": True,
            "random_state": 0,
        }

    @property
    def default_val_opts(self) -> dict:
        return {
            "batch_size": 8,
            "shuffle": False,
            "num_workers": 4,
            "drop_last": False,
            "pin_memory": False,
            "n_splits": 5,
            "kfold_shuffle": True,
            "random_state": 0,
        }


class NetworkConfig(BaseConfig):
    """Network Configuration."""


class LossConfig(BaseConfig):
    """Loss Configuration."""


class SchedulerConfig(BaseConfig):
    """Learning Rate Scheduler Configuration."""


class OptimizerConfig(BaseConfig):
    """Optimizer Configuration."""


class MetricsConfig(BaseConfig):
    """Metrics Configuration."""


def init_transform(
    global_cfg: BaseConfig,
    module: ModuleType = None,
    stage: str = "train",
    *args,
    **kwargs,
) -> Any:

    generic_type = "transform"
    transform_cfg = get_config(generic_type, global_cfg)
    if not module:
        module = get_module_from_type(generic_type)
    return initialize(module, transform_cfg, stage, *args, **kwargs)


def init_dataset(
    global_cfg: BaseConfig,
    module: ModuleType = None,
    stage: str = "train",
    transform_factory=None,
    *args,
    **kwargs,
) -> Any:

    generic_type = "dataset"
    dataset_cfg = get_config(generic_type, global_cfg)

    if not module:
        module = get_module_from_type(generic_type)

    if transform_factory:
        transform = transform_factory.get_transform(stage=stage)
        return initialize(module,
                          dataset_cfg,
                          stage,
                          transform=transform,
                          *args,
                          **kwargs)

    return initialize(module, dataset_cfg, stage, *args, **kwargs)


def init_dataloader(
    global_cfg: BaseConfig,
    dataset,
    module: ModuleType = None,
    stage: str = "train",
    *args,
    **kwargs,
) -> Any:

    generic_type = "dataloader"
    dataloader_cfg = get_config(generic_type, global_cfg)
    if not module:
        module = get_module_from_type(generic_type)
    return initialize(module, dataloader_cfg, stage, dataset, *args, **kwargs)


def get_config(constructor_type: str, global_cfg: BaseConfig) -> BaseConfig:
    """
    Initialize and configure data processing objects.

    Parameters
    ----------
    constructor_type : str
        One of 'transform', 'dataset', 'dataloader', 'network', 'loss',
        'lr_scheduler', 'optimizer', 'metrics'.
    global_cfg : BaseConfig
        Global Configuration object.

    Returns
    -------
    instances : Union[List[Any], Any]
        Configuration object with parameters and options set according to yaml
        configuration file.
    """
    generic_type = get_generic_type(constructor_type)
    if generic_type not in TYPES:
        raise KeyError(f"constructor type must be one of {TYPES}")

    if generic_type not in global_cfg.state:
        msg = "Constructor type missing from provided configuration file"
        raise AttributeError(msg)

    # Ensures a list of recipes
    if not isinstance(global_cfg.get(generic_type), list):
        recipes = [global_cfg.get(generic_type)]
    else:
        recipes = global_cfg.get(generic_type)

    # Get a list of configuration objects from the correct type
    cfg_list = [
        get_config_from_type(generic_type, recipe) for recipe in recipes
    ]
    if len(cfg_list) > 1:
        local_cfg = cfg_list[0]
        for cfg_ in cfg_list[1:]:
            local_cfg.join(cfg_)

    return local_cfg


def initialize(module: ModuleType, cfg: BaseConfig, stage: str, *args,
               **kwargs) -> Any:
    """
    Helper to construct an instance of a class from Configuration object.

    Parameters
    ----------
    module : str
        Module containing the class to construct.
    constructor_type : str
        One of 'transform', 'dataset', 'dataloader', 'network', 'loss',
        'lr_scheduler', 'optimizer', 'metrics'
    cfg : BaseConfig
        Object with the keyword arguments used to construct the class instance.
    stage : str
        Stage, one of 'train' | 'test' | 'val'
    args : list
        Runtime positional arguments used to construct the class instance.
    kwargs : dict
        Runtime keyword arguments used to construct the class instance.

    Returns
    -------
    obj : Any
        Instance of module.
    """
    # Get constructor type
    constructor_type = cfg.type
    # Set stage
    cfg.stage = stage
    # Update options with dinamically defined keywords
    cfg.update(**kwargs)
    cfg_ = cfg.clone()
    argspec = introspect_constructor(constructor_type, module)
    if argspec.keywords is None:
        # Then the class does not support variable keywords
        cfg_.filter(*argspec.defaults.keys())

    obj = get_instance(module, constructor_type, *args, **cfg_.state)

    # Save configuration to cfg attribute. Reset, update and remove defaults
    cfg.reset()
    cfg.update(**kwargs)
    cfg.remove(*argspec.defaults.keys())
    setattr(obj, "cfg", cfg)

    return obj


def get_config_from_type(generic_type: str, options: dict) -> BaseConfig:
    """
    Return the appropriate Configuration instance given type.

    Parameters
    ----------
    generic_type: str
        Name of the object type.
    options : dict
        Options to initialize the Configuration object.

    Returns
    -------
    cfg : BaseConfig
        Configuration object subtype. One of TransformConfig, DatasetConfig,
        DataloaderConfig, NetworkConfig, LossConfig, SchedulerConfig,
        OptimizerConfig, MetricsConfig.
    """

    type_to_config = {
        "transform": TransformConfig,
        "dataset": DatasetConfig,
        "dataloader": DataloaderConfig,
        "network": NetworkConfig,
        "loss": LossConfig,
        "lr_scheduler": SchedulerConfig,
        "optimizer": OptimizerConfig,
        "metrics": MetricsConfig,
    }

    return type_to_config[generic_type](**options)


def get_module_from_type(generic_type: str) -> ModuleType:
    """
    Return the appropriate module name given type.

    Parameters
    ----------
    generic_type: str
        Name of the object type.

    Returns
    -------
    _ : str
        Return the module name. One of DataModule | ModelModule
    """

    type_to_module = {
        "transform": "pytorch_3T27T.data",
        "dataset": "pytorch_3T27T.data",
        "dataloader": "pytorch_3T27T.data",
        "network": "pytorch_3T27T.model",
        "loss": "pytorch_3T27T.model",
        "lr_scheduler": "pytorch_3T27T.model",
        "optimizer": "pytorch_3T27T.model",
        "metrics": "pytorch_3T27T.model",
    }

    mod = importlib.import_module(type_to_module[generic_type])
    return mod


def get_generic_type(given_name: str) -> str:
    """
    Given object name, return the object type according to the global TYPE.

    Parameters
    ----------
    given_name : str
        Given name to the object.

    Returns
    -------
    obj_type : str
        If there is a match, one of TYPE, else returns 'given_name'.
    """

    pattern = "(" + "|".join(TYPES) + ")"
    regex = re.compile(pattern)
    match = regex.search(given_name)
    if match is not None:
        # If there is a match, its in a sigleton tuple
        obj_type = match.groups()[0]
    else:
        obj_type = given_name
    return obj_type


def introspect_constructor(constructor: str, module: ModuleType = None):
    """
    Introspect constructor.

    Parameters
    ----------
    constructor : str
        Name of the object class.

    module : str
        Name of the module where the given object class is defined.

    Returns
    -------
    cfg : Configuration
        Configuration object with parameters and options set according to yaml
        configuration file.
    """

    if module:
        func = getattr(module, constructor, "__init__")
    else:
        func = getattr(constructor, "__init__")
    sig = inspect.signature(func.__init__)
    defaults = {
        p.name: p.default
        for p in sig.parameters.values()
        if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        and p.default is not p.empty
    }
    args = [
        p.name for p in sig.parameters.values() if
        p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and p.name != "self"
    ]
    # Only keep the non default parameters
    args = list(filter(lambda arg: arg not in defaults, args))

    _varargs = [
        p.name for p in sig.parameters.values()
        if p.kind == inspect.Parameter.VAR_POSITIONAL
    ]
    varargs = _varargs[0] if _varargs else None
    _keywords = [
        p.name for p in sig.parameters.values()
        if p.kind == inspect.Parameter.VAR_KEYWORD
    ]
    keywords = _keywords[0] if _keywords else None

    argspec = namedtuple("argspec",
                         ["args", "defaults", "varargs", "keywords"])
    return argspec(args, defaults, varargs, keywords)


def get_instance(module: ModuleType, constructor: str, *args, **kwargs) -> Any:
    """
    Helper to construct an instance of a class.

    Parameters
    ----------
    module : str
        Module containing the class to construct.
    constructor : str
        Name of class, as would be returned by ``.__class__.__name__``.
    args : list
        Positional arguments used to construct the class instance.
    kwargs : dict
        Keyword arguments used to construct the class instance.
    """
    return getattr(module, constructor)(*args, **kwargs)


def retrieve_name(var: Any) -> str:
    """
    Gets the name of var. Does it from the out most frame inner-wards.

    Parameters
    ----------
    var: Any
        Variable to get name from.

    Returns
    -------
    _ : str
        Variable given name
    """
    for live_obj in reversed(inspect.stack()):
        names = [
            var_name for var_name, var_val in live_obj.frame.f_locals.items()
            if var_val is var
        ]
        if len(names) > 0:
            return names[0]
    return ""
