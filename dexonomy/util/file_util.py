from typing import Any
import json
from ruamel.yaml import YAML
import numpy as np
import os
import omegaconf
import traceback
import logging
from functools import wraps

YAML_LOADER = YAML()
YAML_LOADER.allow_duplicate_keys = False


def strip_ruamel(obj: Any) -> Any:
    """Recursively convert ruamel.yaml objects to plain Python types."""
    if hasattr(obj, "items"):  # Covers CommentedMap (dict-like)
        return {k: strip_ruamel(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):  # Covers CommentedSeq (list-like)
        return [strip_ruamel(x) for x in obj]
    else:
        return obj  # int, str, float, etc.


def load_yaml(file_path: str) -> dict:
    with open(file_path) as file_p:
        yaml_param = YAML_LOADER.load(file_p)
    return strip_ruamel(yaml_param)


def load_json(file_path: str) -> dict:
    with open(file_path) as file_p:
        return json.load(file_p)


def write_json(data: dict, file_path: str):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=1)


def _get_template_from_prefix(prefix: str, all_tmpl_names: list[str]) -> str:
    ret_name = None
    for tc in all_tmpl_names:
        if tc.startswith(prefix):
            if ret_name is not None:
                raise ValueError(
                    f"Template name starting with {prefix} is not unique: {ret_name} {tc}"
                )
            ret_name = tc
    assert ret_name is not None, f"Template name starting with {prefix} not found"
    return ret_name


def get_template_names(
    raw_input: omegaconf.listconfig.ListConfig | str | None, init_tmpl_dir: str
) -> list[str]:
    all_tmpl_names = [f.split(".npy")[0] for f in os.listdir(init_tmpl_dir)]
    if raw_input is None:
        tmpl_names = all_tmpl_names
    elif isinstance(raw_input, omegaconf.listconfig.ListConfig):
        tmpl_names = []
        for tn in list(raw_input):
            tmpl_names.append(_get_template_from_prefix(tn, all_tmpl_names))
    elif isinstance(raw_input, str):
        tmpl_names = [_get_template_from_prefix(raw_input, all_tmpl_names)]
    else:
        raise NotImplementedError(f"Unexpected raw input: {raw_input}")
    return tmpl_names


def load_scene_cfg(scene_path: str) -> dict:
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


def _safe_wrapper_func(func, param):
    try:
        return func(param)
    except Exception as e:
        error_traceback = traceback.format_exc()
        logging.warning(f"{error_traceback}")
        return param[0]


def safe_wrapper(func):
    @wraps(func)
    def wrapper(param):
        return _safe_wrapper_func(func, param)

    return wrapper
