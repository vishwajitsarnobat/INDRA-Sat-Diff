# climate_forecast/cli.py

import argparse
import yaml
import os
import sys
from omegaconf import OmegaConf

from .pipelines import preprocess, train, forecast
from .utils.config_loader import load_and_merge_configs

def main():
    """
    The main command-line interface for the Climate Forecasting Framework.
    """
    parser = argparse.ArgumentParser(
        description="A framework for training and running climate forecasting models.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # --- Preprocess and Train Commands ---
    parser_preprocess = subparsers.add_parser('preprocess', help="Preprocess raw data.")
    parser_preprocess.add_argument('--config', required=True, help="Path to the user's YAML configuration file.")

    parser_train = subparsers.add_parser('train', help="Run the full training pipeline.")
    parser_train.add_argument('--config', required=True, help="Path to the user's YAML configuration file.")

    # --- Forecast Command (Mirrors user's main.py) ---
    parser_forecast = subparsers.add_parser('forecast', help="Generate a forecast from a trained model.")
    parser_forecast.add_argument('--config', required=True, help="Path to the user's main configuration file.")
    parser_forecast.add_argument('--input', required=True, dest='input_file',
                                 help="Path to the LAST file of the input sequence (the T=0 frame).")
    parser_forecast.add_argument('--output-dir', dest='output_dir', default="output",
                                 help="Directory for forecast results. Defaults to './output'.")
    args = parser.parse_args()

    # --- Execute Command ---
    if args.command == 'preprocess':
        try:
            with open(args.config, 'r') as f:
                user_config = yaml.safe_load(f)
            preprocess.run(config=user_config)
        except Exception as e:
            print(f"Error during preprocessing: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == 'train':
        try:
            full_config = load_and_merge_configs(user_config_path=args.config, default_config_name='train.yaml')
            train.run(config=full_config)
        except Exception as e:
            print(f"Error during training: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == 'forecast':
        try:
            full_config = load_and_merge_configs(user_config_path=args.config, default_config_name='train.yaml')
            
            full_config.setdefault('forecast', {})
            full_config['forecast']['input_file'] = args.input_file
            full_config['forecast']['output_dir'] = args.output_dir
            
            forecast.run(config=full_config)
        except Exception as e:
            print(f"Error during forecasting: {e}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()