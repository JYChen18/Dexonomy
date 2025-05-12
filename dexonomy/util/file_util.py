from typing import Dict, Union
import json
from ruamel.yaml import YAML

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


def write_yaml(data: Dict, file_path: str):
    """Write dictionary to yaml file.

    Args:
        data: Dictionary to write to yaml file.
        file_path: Path to write the yaml file.
    """
    with open(file_path, "w") as file:
        yaml.dump(data, file)


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
