# climate_forecast/utils/config_loader.py

import yaml
from importlib import resources
import collections.abc

def _deep_merge_dicts(base, update):
    """
    Recursively merges dictionaries.
    
    Keys from `update` will overwrite keys from `base`. If a key exists
    in both and the values are dictionaries, it will recursively merge them.
    """
    for key, value in update.items():
        if isinstance(value, collections.abc.Mapping) and key in base and isinstance(base[key], collections.abc.Mapping):
            base[key] = _deep_merge_dicts(base[key], value)
        else:
            base[key] = value
    return base

def load_and_merge_configs(user_config_path: str, default_config_name: str) -> dict:
    """
    Loads a default config from within the package and merges a user's config on top.

    This is the core function for making the framework configurable with defaults.

    Args:
        user_config_path (str): Path to the user's YAML config file.
        default_config_name (str): Filename of the default config template
                                   (e.g., 'train.yaml').

    Returns:
        A single, merged dictionary with user values overriding defaults.
    """
    try:
        # --- 1. Load the default template from within the package ---
        # `importlib.resources` is the modern, correct way to access package data.
        with resources.files('climate_forecast').joinpath(f'configs/{default_config_name}').open('r') as f:
            default_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Default configuration template '{default_config_name}' not found within the package.")
        raise
    except Exception as e:
        print(f"Error reading default configuration template: {e}")
        raise

    try:
        # --- 2. Load the user's config file ---
        with open(user_config_path, 'r') as f:
            user_config = yaml.safe_load(f)
        if user_config is None:
            user_config = {} # Treat an empty file as an empty dictionary
    except FileNotFoundError:
        print(f"Error: User configuration file not found at: {user_config_path}")
        raise
    except Exception as e:
        print(f"Error reading user configuration file: {e}")
        raise

    # --- 3. Perform a deep merge ---
    # The user's config is the 'update', so its values take precedence.
    level = user_config.get('compute_level', default_config.get('compute_level', 1))
    default_config = apply_compute_level(default_config, level)
    merged_config = _deep_merge_dicts(default_config, user_config)
    merged_config['compute_level'] = level

    return merged_config


VALID_LEVELS = (1, 2, 4, 8, 16)


def load_default_config(name: str = 'train.yaml') -> dict:
    with resources.files('climate_forecast').joinpath(f'configs/{name}').open('r') as f:
        return yaml.safe_load(f)


def load_level_presets() -> dict:
    with resources.files('climate_forecast').joinpath('configs/levels.yaml').open('r') as f:
        return {int(k): v for k, v in (yaml.safe_load(f) or {}).items()}


def apply_compute_level(config: dict, level) -> dict:
    """Merge the preset for `level` into `config` (used on the defaults, so user values still win)."""
    level = int(level)
    if level not in VALID_LEVELS:
        raise ValueError(f"compute_level must be one of {VALID_LEVELS}, got {level}")
    if level == 1:
        return config
    import copy
    return _deep_merge_dicts(copy.deepcopy(config), copy.deepcopy(load_level_presets()[level]))