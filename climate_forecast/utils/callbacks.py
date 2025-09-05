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
        
        # Not working properly, might improve in future
        # if stage_name == 'indra_sat_diff':
        #     self._plot_skill_scores(history, save_dir)

    def _plot_loss_curve(self, history, save_dir, stage_name):
        """
        Plots the main training and validation loss for a given stage. This version robustly
        finds the correct metric keys for all stages.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        epochs = [d.get('epoch', i) for i, d in enumerate(history)]

        # Potential keys for training loss (covers VAE, Alignment, and IndraSatDiff stages)
        train_keys = ['train/loss_epoch', 'train/total_loss_epoch']
        # Potential keys for validation loss
        val_keys = ['val/loss_epoch', 'val/loss']

        # Find the first available key from the list that exists in the logged metrics
        train_key_found = next((key for key in train_keys if key in history[0]), None)
        val_key_found = next((key for key in val_keys if key in history[0]), None)

        # Plot Training Loss if a valid key was found
        if train_key_found:
            values = [d.get(train_key_found) for d in history]
            valid_epochs = [e for e, v in zip(epochs, values) if v is not None]
            valid_values = [v for v in values if v is not None]
            if valid_values:
                ax.plot(valid_epochs, valid_values, marker='o', linestyle='-', label='Training Loss')

        # Plot Validation Loss if a valid key was found
        if val_key_found:
            values = [d.get(val_key_found) for d in history]
            valid_epochs = [e for e, v in zip(epochs, values) if v is not None]
            valid_values = [v for v in values if v is not None]
            if valid_values:
                ax.plot(valid_epochs, valid_values, marker='o', linestyle='-', label='Validation Loss')

        ax.set_xlabel("Epoch", fontsize=16)
        ax.set_ylabel("Loss Value", fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_title(f"{stage_name.replace('_', ' ').title()} Stage: Training & Validation Loss", fontsize=18, fontweight='bold')
        
        if train_key_found or val_key_found:
            ax.legend(fontsize=14)

        ax.grid(True, linestyle='--', alpha=0.6)
        
        if epochs:
            max_epoch = max(epochs)
            tick_step = max(1, (max_epoch + 1) // 10)
            ax.set_xticks(range(0, max_epoch + 1, tick_step))
        
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "loss_plot.png"), dpi=150)
        plt.close(fig)

    def _plot_skill_scores(self, history, save_dir):
        """
        Plots the primary skill scores (CSI, BIAS, FSS) for the final model.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        epochs = [d.get('epoch', i) for i, d in enumerate(history)]

        skill_scores_to_plot = {
            'Average CSI': 'val/csi_avg_epoch',
            'Average BIAS': 'val/bias_avg_epoch',
            'Average FSS': 'val/fss_avg_epoch',
        }

        plotted_anything = False
        for label, key in skill_scores_to_plot.items():
            if key in history[0]:
                plotted_anything = True
                values = [d.get(key) for d in history]
                valid_epochs = [e for e, v in zip(epochs, values) if v is not None]
                valid_values = [v for v in values if v is not None]
                if valid_values:
                    ax.plot(valid_epochs, valid_values, marker='o', linestyle='-', label=label)

        ax.set_xlabel("Epoch", fontsize=16)
        ax.set_ylabel("Skill Score", fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_title("INDRA-Sat-Diff Stage: Validation Skill Scores", fontsize=18, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)
        
        if epochs:
            max_epoch = max(epochs)
            tick_step = max(1, (max_epoch + 1) // 10)
            ax.set_xticks(range(0, max_epoch + 1, tick_step))

        if 'val/bias_avg_epoch' in history[0]:
            ax.axhline(y=1.0, color='r', linestyle='--', linewidth=1.5, label='Ideal BIAS = 1.0')
        
        if plotted_anything:
            ax.legend(fontsize=14)

        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "skill_scores_plot.png"), dpi=150)
        plt.close(fig)

    def _print_summary_table(self, epoch_metrics, stage_name):
        if not RICH_AVAILABLE: return
        
        table = Table(title=f"End of Epoch {epoch_metrics.get('epoch')} | Stage: {stage_name.upper()}")
        table.add_column("Metric", justify="right", style="cyan", no_wrap=True)
        table.add_column("Value", justify="left", style="magenta")
        
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