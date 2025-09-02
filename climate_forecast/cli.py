import argparse
import yaml
import os
import sys

# Import the high-level pipeline functions from our framework
from climate_forecast.pipelines import preprocess, train, forecast
from climate_forecast.utils.config_loader import load_and_merge_configs

def main():
    """
    The main command-line interface for the Climate Forecasting Framework.
    
    This function is executed when the user runs `climate-forecast` from their terminal.
    """
    # Main parser
    parser = argparse.ArgumentParser(
        description="Climate Forecasting Framework CLI",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # --- Preprocess Command ---
    parser_preprocess = subparsers.add_parser('preprocess', help="Preprocess raw data into the framework's format.")
    parser_preprocess.add_argument('--config', required=True, help="Path to the user's YAML configuration file.")

    # --- Train Command ---
    parser_train = subparsers.add_parser('train', help="Run the full training pipeline (VAE -> Alignment -> PreDiff).")
    parser_train.add_argument('--config', required=True, help="Path to the user's YAML configuration file.")

    # --- Forecast Command ---
    parser_forecast = subparsers.add_parser('forecast', help="Generate a forecast using a trained model.")
    parser_forecast.add_argument('--config', required=True, help="Path to the user's YAML configuration file.")

    args = parser.parse_args()

    # --- Load User Configuration ---
    try:
        with open(args.config, 'r') as f:
            user_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{os.path.abspath(args.config)}'", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading YAML configuration from '{args.config}': {e}", file=sys.stderr)
        sys.exit(1)

    # --- Execute Command ---
    if args.command == 'preprocess':
        # For preprocessing, we only need the user's config.
        preprocess.run(config=user_config)
        
    elif args.command == 'train':
        # For training, we merge the user's config with our default template.
        full_config = load_and_merge_configs(
            user_config_path=args.config,
            default_config_name='train.yaml'
        )
        train.run(config=full_config)
        
    elif args.command == 'forecast':
        # Forecasting also uses a merged config (though it has fewer defaults).
        full_config = load_and_merge_configs(
            user_config_path=args.config,
            default_config_name='forecast.yaml' # We should create this minimal template too
        )
        forecast.run(config=full_config)

if __name__ == "__main__":
    main()