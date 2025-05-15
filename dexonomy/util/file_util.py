from typing import Dict, Union
import json
from ruamel.yaml import YAML
import numpy as np
import os
import omegaconf

YAML_LOADER = YAML()
YAML_LOADER.allow_duplicate_keys = False


def strip_ruamel(obj):
    """Recursively convert ruamel.yaml objects to plain Python types."""
    if hasattr(obj, "items"):  # Covers CommentedMap (dict-like)
        return {k: strip_ruamel(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):  # Covers CommentedSeq (list-like)
        return [strip_ruamel(x) for x in obj]
    else:
        return obj  # int, str, float, etc.


def load_yaml(file_path: Union[str, Dict]) -> Dict:
    """Load yaml file and return as dictionary. If file_path is a dictionary, return as is.

    Args:
        file_path: File path to yaml file or dictionary.

    Returns:
        Dict: Dictionary containing yaml file content.
    """
    if isinstance(file_path, str):
        with open(file_path) as file_p:
            yaml_params = YAML_LOADER.load(file_p)
        yaml_params = strip_ruamel(yaml_params)
    else:
        yaml_params = file_path
    return yaml_params


def load_json(file_path):
    if isinstance(file_path, str):
        with open(file_path) as file_p:
            json_params = json.load(file_p)
    else:
        json_params = file_path
    return json_params


def write_json(data: Dict, file_path):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=1)


def get_template_name_lst(template_name, init_template_dir):
    all_template_lst = [f.split(".npy")[0] for f in os.listdir(init_template_dir)]
    if template_name is None:
        hand_template_names = all_template_lst
    elif isinstance(template_name, omegaconf.listconfig.ListConfig):
        hand_template_names = list(template_name)
        for tn in hand_template_names:
            assert tn in all_template_lst
    elif isinstance(template_name, str):
        assert template_name in all_template_lst
        hand_template_names = [template_name]
    else:
        raise NotImplementedError(f"Undefined type of template_name: {template_name}")
    return hand_template_names


def load_scene_cfg(scene_path):
    scene_cfg = np.load(scene_path, allow_pickle=True).item()

    def update_relative_path(d: dict):
        for k, v in d.items():
            if isinstance(v, dict):
                update_relative_path(v)
            elif k.endswith("_path") and isinstance(v, str):
                d[k] = os.path.join(os.path.dirname(scene_path), v)
        return

    update_relative_path(scene_cfg["scene"])

    return scene_cfg
