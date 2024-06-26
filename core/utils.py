import subprocess
import os
from core.data import uniform_01_generator, uniform_23_generator, uniform_416_47_generator

# Clipping valuation values functions
from core.clip_ops.clip_ops import *

# Data generator classes
from core.data import *


CLIPS = {
    'uniform_01': lambda x: clip_op_01(x),
    'uniform_12': lambda x: clip_op_12(x),
    'uniform_23': lambda x: clip_op_23(x),
    'uniform_416_47': lambda x: clip_op_416_47(x)
}

GENERATORS = {
    'uniform_01': uniform_01_generator.Generator,
    'uniform_416_47': uniform_416_47_generator.Generator,
    # 'uniform_12': uniform_12_generator.Generator,
    'uniform_23': uniform_23_generator.Generator
}


def get_path_and_file(setting_path):
    path, file = os.path.split(setting_path)
    return path, os.path.splitext(file)[0]


def get_objects(setting_path):
    '''
    Get objects from configuration file
    '''
    path, setting_name = get_path_and_file(setting_path)
    print(f"PATH: {path}")
    print(f"SETTING NAME: {setting_name}")
    import_obj = __import__(path, fromlist=[setting_name])
    cfg = getattr(import_obj, setting_name).cfg
    print(f"CFG: {cfg}")
    clip_op = CLIPS[cfg.distribution_type]
    print(f"CLIP OP: {clip_op}")
    generator = GENERATORS[cfg.distribution_type]
    # generator.save_data(iter =0)
    print(f"GENERATOR: {generator}")
    print()
    return cfg, clip_op, generator, setting_name


def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"], encoding="utf-8"
    )
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split("\n")]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map
