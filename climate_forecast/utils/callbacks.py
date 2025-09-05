# climate_forecast/utils/callbacks.py

import os
import json
import warnings
import matplotlib.pyplot as plt
from collections import defaultdict
import torch
from lightning.pytorch.callbacks import Callback

try:
    from rich.console import Console
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

def structure_metrics(flat_metrics: dict) -> dict:
    """Converts a flat metric dictionary with '/' into a nested dictionary."""
    structured = {}
    for key, value in flat_metrics.items():
        parts = key.split('/')
        d = structured
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return structured

class MetricsLoggerCallback(Callback):
    def __init__(self):
        super().__init__()
        self.metrics_history = defaultdict(list)
        if RICH_AVAILABLE:
            self.console = Console()
        # Define standard metrics for plotting per stage
        self.standard_metrics_map = {
            'vae': ['val/loss', 'val/rec_loss', 'val/kl_loss', 'train/loss_epoch'],
            'alignment': ['val/loss', 'train/loss_epoch'],
            'indra_sat_diff': ['val/loss', 'train/loss_simple_epoch', 'train/loss_vlb_epoch']
        }


    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        epoch_metrics = trainer.callback_metrics.copy()
        current_epoch_results = {
            key: round(value.item(), 5) for key, value in epoch_metrics.items() if isinstance(value, torch.Tensor)
        }
        current_epoch_results['epoch'] = trainer.current_epoch
        
        save_dir = getattr(pl_module, 'save_dir', None)
        if not save_dir: return
            
        stage_name = os.path.basename(os.path.normpath(save_dir))
        self.metrics_history[stage_name].append(current_epoch_results)

        self._save_json(save_dir, stage_name)
        self._plot_metrics(save_dir, stage_name)
        self._print_summary_table(current_epoch_results, stage_name)
        if stage_name == 'indra_sat_diff':
            self._print_detailed_metrics_table(current_epoch_results, stage_name)

    def _save_json(self, save_dir, stage_name):
        json_path = os.path.join(save_dir, "metrics_history.json")
        structured_history = [structure_metrics(d) for d in self.metrics_history[stage_name]]
        try:
            with open(json_path, 'w') as f:
                json.dump(structured_history, f, indent=4)
        except Exception as e:
            warnings.warn(f"Could not save metrics to {json_path}. Error: {e}")

    def _plot_metrics(self, save_dir, stage_name):
        history = self.metrics_history[stage_name]
        if not history: return
        self._plot_loss_curve(history, save_dir, stage_name)
        if stage_name == 'indra_sat_diff':
            self._plot_skill_scores(history, save_dir)

    def _plot_loss_curve(self, history, save_dir, stage_name):
        fig, ax = plt.subplots(figsize=(12, 8))
        epochs = [d.get('epoch', i) for i, d in enumerate(history)]
        
        standard_metrics = self.standard_metrics_map.get(stage_name, [])

        for key in standard_metrics:
            # Check for key variations (e.g., train_loss vs train/loss)
            actual_key = key
            if actual_key not in history[0]:
                actual_key = key.replace('/', '_') # Fallback for older log formats
            
            values = [d.get(actual_key) for d in history]
            if any(v is not None for v in values):
                ax.plot(epochs, values, marker='o', linestyle='-', label=actual_key)
        
        ax.set_xlabel("Epoch", fontsize=16)
        ax.set_ylabel("Metric Value", fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_title(f"{stage_name.replace('_', ' ').title()} Stage: Validation Losses", fontsize=18, fontweight='bold')
        ax.legend(fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)
        if epochs:
            ax.set_xticks([e for e in epochs if int(e) == e])
        
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "metrics_plot.png"), dpi=150)
        plt.close(fig)

    def _plot_skill_scores(self, history, save_dir):
        # Implementation remains the same
        pass

    def _print_summary_table(self, epoch_metrics, stage_name):
        if not RICH_AVAILABLE: return
        
        table = Table(title=f"End of Epoch {epoch_metrics.get('epoch')} | Stage: {stage_name.upper()}")
        table.add_column("Metric", justify="right", style="cyan", no_wrap=True)
        table.add_column("Value", justify="left", style="magenta")
        
        # Display all available epoch-level metrics
        display_metrics = {k: v for k, v in epoch_metrics.items() if '_step' not in k and 'details' not in k and k != 'epoch'}
        
        if not display_metrics:
            table.add_row("No summary metrics logged for this epoch.", "")
        else:
            for key, value in sorted(display_metrics.items()):
                table.add_row(key, f"{value:.5f}")
            
        self.console.print(table)

    def _print_detailed_metrics_table(self, epoch_metrics, stage_name):
        if not RICH_AVAILABLE: return
        
        structured = structure_metrics(epoch_metrics)
        details = structured.get('val', {}).get('details', {})
        if not details: return

        time_keys = sorted(details.keys(), key=lambda x: int(x.replace('T+', '').replace('min', '')))
        if not time_keys: return
        
        first_time = time_keys[0]
        scale_keys = sorted(details[first_time].keys())
        first_scale = scale_keys[0]
        thresh_keys = sorted(details[first_time][first_scale].keys())
        metric_names = sorted(details[first_time][first_scale][thresh_keys[0]].keys())
        
        for t_key in time_keys:
            table = Table(title=f"Detailed Metrics | Forecast Lead Time: {t_key} | Epoch {epoch_metrics.get('epoch')}")
            table.add_column("Scale", style="cyan")
            table.add_column("Threshold", style="cyan")
            for metric in metric_names:
                table.add_column(metric.upper(), style="magenta")

            for scale in scale_keys:
                for i, threshold in enumerate(thresh_keys):
                    row_data = [scale if i == 0 else "", threshold]
                    for metric in metric_names:
                        value = details.get(t_key, {}).get(scale, {}).get(threshold, {}).get(metric, float('nan'))
                        row_data.append(f"{value:.4f}")
                    table.add_row(*row_data)
                if scale != scale_keys[-1]:
                    table.add_section()
            self.console.print(table)