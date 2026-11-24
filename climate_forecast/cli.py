# climate_forecast/cli.py

import argparse
import yaml
import os
import sys
from omegaconf import OmegaConf

from .pipelines import preprocess, train, forecast, download
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

    # --- Download Command ---
    parser_download = subparsers.add_parser('download', help="Download IMERG data for a region and period.")
    parser_download.add_argument('--config', required=True, help="YAML config (uses its 'download' and 'preprocess' sections).")
    parser_download.add_argument('--start', help="UTC start, e.g. 2023-12-01T00:00 (overrides config).")
    parser_download.add_argument('--end', help="UTC end, inclusive, e.g. 2023-12-06T23:30 (overrides config).")
    parser_download.add_argument('--output-dir', dest='output_dir', help="Where raw files go (default: preprocess.input_dir).")

    parser_fit = subparsers.add_parser('fit-check', help="Measure GPU memory per compute level (random data).")
    parser_fit.add_argument('--config', required=True, help="Path to the user's YAML configuration file.")
    parser_fit.add_argument('--levels', nargs='+', type=int, help="Levels to test (default: 1 2 4 8 16).")
    parser_fit.add_argument('--steps', type=int, default=3, help="Training steps per stage (default: 3).")

    # --- Preprocess and Train Commands ---
    parser_preprocess = subparsers.add_parser('preprocess', help="Preprocess raw data.")
    parser_preprocess.add_argument('--config', required=True, help="Path to the user's YAML configuration file.")

    parser_train = subparsers.add_parser('train', help="Run the full training pipeline.")
    parser_train.add_argument('--config', required=True, help="Path to the user's YAML configuration file.")

    # --- Forecast Command ---
    parser_forecast = subparsers.add_parser('forecast', help="Generate a forecast from a trained model.")
    parser_forecast.add_argument('--config', required=True, help="Path to the user's main configuration file.")
    parser_forecast.add_argument('--input', required=True, dest='input_file',
                                 help="Path to the LAST file of the input sequence (the T=0 frame).")
    parser_forecast.add_argument('--output-dir', '--output', dest='output_dir', default="output",
                                 help="Directory for forecast results. Defaults to './output'.")
    parser_forecast.add_argument('--precision', choices=['fp32', 'fp16', 'bf16'],
                                 help="Inference precision; fp16/bf16 roughly halve GPU memory (default fp32).")
    parser_forecast.add_argument('--sampler', choices=['ddim', 'ddpm'], help="ddim (fast, default) or ddpm (all training steps).")
    parser_forecast.add_argument('--steps', type=int, help="DDIM steps (default from config: 20).")
    parser_forecast.add_argument('--ensemble', type=int, help="Number of ensemble members (default from config: 4).")
    parser_forecast.add_argument('--postprocess', choices=['member', 'mean', 'prob', 'pm', 'pmi'],
                                 help="How the ensemble becomes one forecast (default from config: pmi).")
    parser_forecast.add_argument('--ka-multiplier', type=float, help="KA target = this x mean input intensity (default 1.0).")
    parser_forecast.add_argument('--no-alignment', dest='use_alignment', action='store_false',
                                 help="Disable knowledge alignment guidance during sampling.")
    args = parser.parse_args()

    # --- Execute Command ---
    if args.command == 'download':
        try:
            with open(args.config, 'r') as f:
                user_config = yaml.safe_load(f)
            d = user_config.setdefault('download', {})
            if args.start or args.end:
                if not (args.start and args.end):
                    raise ValueError("Pass both --start and --end.")
                d['start'], d['end'] = args.start, args.end
                d.pop('periods', None)
            if args.output_dir:
                d['output_dir'] = args.output_dir
            download.run(config=user_config)
        except Exception as e:
            print(f"Error during download: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == 'fit-check':
        from .pipelines import fit_check
        from .utils.config_loader import load_default_config
        with open(args.config, 'r') as f:
            user_config = yaml.safe_load(f) or {}
        fit_check.run(user_config, load_default_config('train.yaml'), levels=args.levels, steps=args.steps)

    elif args.command == 'preprocess':
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
            if not args.use_alignment:
                full_config['forecast']['use_alignment'] = False
            if args.precision:
                full_config['forecast']['precision'] = args.precision
            for key, val in (('sampler', args.sampler), ('steps', args.steps), ('ensemble_size', args.ensemble),
                             ('postprocess', args.postprocess), ('ka_multiplier', args.ka_multiplier)):
                if val is not None:
                    full_config['forecast'][key] = val
            
            forecast.run(config=full_config)
        except Exception as e:
            print(f"Error during forecasting: {e}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()