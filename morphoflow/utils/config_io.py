import argparse
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / 'configs'
TRAIN_CONFIG_DEFAULT = CONFIG_DIR / 'train_config_default.yml'
EVAL_CONFIG_DEFAULT = CONFIG_DIR / 'eval_config_default.yml'


def write_config(source_config_file: Path, target_dir: Path) -> None:
    """Copy the source config file into the target directory."""
    if not target_dir.exists():
        raise ValueError(f'Target directory for writing config does not exist:\n{target_dir}')
    shutil.copy(str(source_config_file), str(target_dir / source_config_file.name))


def read_yaml_config(config_file_path: Union[Path, str],
                     default_config_file_path: Optional[Path] = None) -> Dict:
    """Read a YAML config, on top of an optional config with default values."""
    config_file_path = Path(config_file_path)
    if not config_file_path.exists():
        raise ValueError(f'YAML config file does not exist:\n{config_file_path}')
    params = {}
    if default_config_file_path is not None:
        with open(default_config_file_path, 'r') as f:
            params.update(yaml.safe_load(f))
    with open(config_file_path, 'r') as f:
        params.update(yaml.safe_load(f) or {})
    return params


def read_path_config(config_file_path: Union[Path, str]) -> Dict:
    """Read the paths config; every *_basedir entry becomes a Path."""
    params = read_yaml_config(config_file_path)
    for key in params:
        if key.endswith('_basedir'):
            params[key] = Path(params[key])
    return params


def parse_config_train() -> Tuple[Dict, Path]:
    """For training: merge the paths config with the training config (on top of the defaults).

    :return: the merged parameters and path to the training config file.
    """
    parser = argparse.ArgumentParser(description='Train MorphoFlow.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--path_file', type=Path, required=True,
                        help='Paths config (data, case files, model and output folders).')
    parser.add_argument('-c', '--config_file', type=Path, required=True,
                        help='Training config.')
    args = parser.parse_args()

    params = read_path_config(args.path_file)
    params.update(read_yaml_config(args.config_file, TRAIN_CONFIG_DEFAULT))
    return params, args.config_file


def parse_config_eval() -> Dict:
    """For inference: merge the paths config, the training config stored with the model and the
    evaluation config (each later one takes precedence).
    """
    parser = argparse.ArgumentParser(description='Evaluate a trained MorphoFlow model.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--path_file', type=Path, required=True,
                        help='Paths config (data, case files, model and output folders).')
    parser.add_argument('-m', '--model_name', type=str, required=True,
                        help='Name of the model folder inside model_basedir.')
    parser.add_argument('-c', '--eval_config', type=Path, required=True,
                        help='Evaluation config.')
    args = parser.parse_args()

    params = read_path_config(args.path_file)
    model_dir = params['model_basedir'] / args.model_name
    model_configs = list(model_dir.glob('*.yml'))
    if len(model_configs) != 1:
        raise ValueError(f'The model directory must contain exactly one training config '
                         f'(*.yml):\n{model_dir}')
    params.update(read_yaml_config(model_configs[0], TRAIN_CONFIG_DEFAULT))
    params.update(read_yaml_config(args.eval_config, EVAL_CONFIG_DEFAULT))

    # The model name in the config file MUST coincide with the directory name!
    if params['model_name'] != args.model_name:
        raise ValueError(f'Cannot load stored model "{args.model_name}": it contains a different '
                         f'model name in the config: "{params["model_name"]}".')
    return params
