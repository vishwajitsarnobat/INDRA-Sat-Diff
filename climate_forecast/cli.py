# climate_forecast/cli.py

import argparse
import yaml
import os
import sys
from omegaconf import OmegaConf

# Import the high-level pipeline functions from our framework
from .pipelines import preprocess, train, forecast
from .utils.config_loader import load_and_merge_configs

def main():
    """
    The main command-line interface for the Climate Forecasting Framework.
    This function is executed when the user runs `climate-forecast` from their terminal.
    """
    parser = argparse.ArgumentParser(
        description="A framework for training and running climate forecasting models.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # --- Preprocess Command ---
    parser_preprocess = subparsers.add_parser(
        'preprocess',
        help="Preprocess raw data into the framework's HDF5 format."
    )
    parser_preprocess.add_argument(
        '--config', required=True,
        help="Path to the user's YAML configuration file."
    )

    # --- Train Command ---
    parser_train = subparsers.add_parser(
        'train',
        help="Run the full training pipeline (VAE -> Alignment -> PreDiff)."
    )
    parser_train.add_argument(
        '--config', required=True,
        help="Path to the user's YAML configuration file."
    )

    # --- Forecast Command ---
    parser_forecast = subparsers.add_parser(
        'forecast',
        help="Generate a forecast using a trained model."
    )
    parser_forecast.add_argument(
        '--config', required=True,
        help="Path to the user's original training YAML configuration file."
    )
    parser_forecast.add_argument(
        '--input', required=True, dest='input_file',
        help="Path to the FIRST file of the input sequence (e.g., the T-N frame)."
    )
    parser_forecast.add_argument(
        '--output-dir', required=True, dest='output_dir',
        help="Directory where the forecast outputs (GIF, NPZ) will be saved."
    )
    parser_forecast.add_argument(
        '--ckpt', required=True, dest='prediff_checkpoint_path',
        help="Path to the final trained model checkpoint file (prediff_final.pt)."
    )

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
            # For training, we merge the user's config with our default template.
            full_config = load_and_merge_configs(
                user_config_path=args.config,
                default_config_name='train.yaml'
            )
            train.run(config=full_config)
        except Exception as e:
            print(f"Error during training: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == 'forecast':
        try:
            # Forecasting uses the user's *training* config to know the model architecture,
            # and we inject the CLI arguments for the specific forecast run.
            full_config = load_and_merge_configs(
                user_config_path=args.config,
                default_config_name='train.yaml' # Use train defaults as base
            )
            
            # Create a 'forecast' section and populate it from CLI args
            forecast_overrides = {
                'forecast': {
                    'input_file': args.input_file,
                    'output_dir': args.output_dir,
                    'prediff_checkpoint_path': args.prediff_checkpoint_path
                }
            }
            # Use OmegaConf to cleanly merge the CLI args into the config
            cfg = OmegaConf.create(full_config)
            cfg.merge_with(OmegaConf.create(forecast_overrides))
            
            # Convert back to a plain dictionary for the pipeline
            final_config = OmegaConf.to_container(cfg, resolve=True)

            forecast.run(config=final_config)
        except Exception as e:
            print(f"Error during forecasting: {e}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()