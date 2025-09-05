# examples/main.py

import argparse
import yaml
import os
import sys
import traceback

# This assumes the 'climate_forecast' package is installed in the environment.
from climate_forecast.pipelines import preprocess, train, forecast
from climate_forecast.utils.config_loader import load_and_merge_configs

def main():
    """
    Command-Line Interface for the Climate Forecasting Framework.
    This script provides three main commands:
    1. preprocess: Converts raw data into a model-ready format.
    2. train: Runs the complete three-stage training pipeline.
    3. forecast: Generates a forecast using a trained model.
    """
    parser = argparse.ArgumentParser(
        description="Climate Forecasting Framework CLI",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # --- Preprocess Command ---
    parser_preprocess = subparsers.add_parser('preprocess', help="Preprocess raw climate data.")
    parser_preprocess.add_argument(
        '--config', default="config.yaml", help="Path to the user's YAML configuration file (default: config.yaml)."
    )

    # --- Train Command ---
    parser_train = subparsers.add_parser('train', help="Run the full model training pipeline.")
    parser_train.add_argument(
        '--config', default="config.yaml", help="Path to the user's YAML configuration file (default: config.yaml)."
    )

    # --- Forecast Command ---
    parser_forecast = subparsers.add_parser('forecast', help="Generate a forecast from a trained model.")
    parser_forecast.add_argument(
        '--config', default="config.yaml", help="Path to the user's main configuration file (default: config.yaml)."
    )
    parser_forecast.add_argument(
        '--input', required=True, dest='input_file',
        help="Path to the LAST .HDF5 file of the input sequence (the T=0 frame)."
    )
    parser_forecast.add_argument(
        '--output', required=False, dest='output_dir', default="output",
        help="Directory for forecast results (default: './output')."
    )

    args = parser.parse_args()

    # --- Execute Command ---
    try:
        if args.command == 'preprocess':
            print("--- Loading configuration for Preprocessing ---")
            with open(args.config, 'r') as f:
                user_config = yaml.safe_load(f)
            preprocess.run(config=user_config)
        
        elif args.command == 'train':
            print("--- Loading and merging configurations for Training ---")
            # Loads the framework's default 'train.yaml' and merges the user's config on top
            full_config = load_and_merge_configs(user_config_path=args.config, default_config_name='train.yaml')
            train.run(config=full_config)
        
        elif args.command == 'forecast':
            print("--- Loading and merging configurations for Forecasting ---")
            # The forecast run needs the full configuration context
            full_config = load_and_merge_configs(user_config_path=args.config, default_config_name='train.yaml')
            
            # Inject the runtime CLI arguments into the config dictionary
            full_config.setdefault('forecast', {})
            full_config['forecast']['input_file'] = args.input_file
            full_config['forecast']['output_dir'] = args.output_dir
            
            forecast.run(config=full_config)

    except Exception as e:
        print(f"\n--- A critical error occurred during the '{args.command}' stage ---", file=sys.stderr)
        print(f"Error: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()