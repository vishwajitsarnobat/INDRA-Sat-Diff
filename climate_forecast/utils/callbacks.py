# climate_forecast/utils/callbacks.py

import os
import json
import warnings
import matplotlib.pyplot as plt
from collections import defaultdict
from lightning.pytorch.callbacks import Callback

class MetricsLoggerCallback(Callback):
    """
    A PyTorch Lightning Callback to intelligently log, plot, and save metrics.

    This callback automatically discovers metrics logged by the LightningModule,
    groups them by keywords (e.g., 'loss', 'csi', 'mae'), and plots them in
    separate subplots. It runs at the end of each validation epoch.
    """
    def __init__(self):
        super().__init__()
        self.metrics_history = []
        # Define keywords to group metrics for plotting. The order determines plot order.
        self.plot_groups = {
            'Loss': ['loss'],
            'CSI Skill Score': ['csi'],
            'Mean Absolute Error': ['mae'],
            'Mean Squared Error': ['mse'],
            'BIAS Score': ['bias'],
            'POD Score': ['pod'],
            'SUCR Score': ['sucr'],
        }

    def on_validation_epoch_end(self, trainer, pl_module):
        # The `callback_metrics` dictionary contains all values logged with self.log()
        # for the current epoch.
        epoch_metrics = trainer.callback_metrics
        
        # --- 1. Clean and Prepare Metrics for Saving ---
        current_epoch_results = {
            key: round(value.item(), 5) if hasattr(value, 'item') else value
            for key, value in epoch_metrics.items()
        }
        current_epoch_results['epoch'] = trainer.current_epoch
        self.metrics_history.append(current_epoch_results)
        
        # The save_dir is taken from the lightning module
        save_dir = getattr(pl_module, 'save_dir', None)
        if not save_dir:
            warnings.warn("pl_module.save_dir is not set. Cannot save metrics JSON or plot.")
            return

        # --- 2. Save All Metrics to JSON ---
        json_path = os.path.join(save_dir, "metrics_history.json")
        try:
            with open(json_path, 'w') as f:
                json.dump(self.metrics_history, f, indent=4)
        except Exception as e:
            warnings.warn(f"Could not save metrics to {json_path}. Error: {e}")

        # --- 3. Plot Key Metrics ---
        self._plot_metrics(save_dir)

    def _plot_metrics(self, save_dir):
        # Dynamically discover which metrics to plot based on what's available
        # in the history and our predefined keywords.
        
        # A dictionary to hold the data for each plot: { 'Plot Title': {'Metric Name': [values]} }
        grouped_metrics_data = defaultdict(lambda: defaultdict(list))
        
        # Find all available metric names from the entire history
        all_keys = set()
        for epoch_data in self.metrics_history:
            all_keys.update(epoch_data.keys())
            
        # Group the available metric keys
        for key in sorted(list(all_keys)):
            for title, keywords in self.plot_groups.items():
                if any(keyword in key for keyword in keywords):
                    # We only want to plot epoch-level summaries
                    if 'epoch' in key:
                        grouped_metrics_data[title][key] = []


        if not grouped_metrics_data:
            # No plottable metrics were found
            return

        # Populate the data for the plots
        epochs = [d.get('epoch', i) for i, d in enumerate(self.metrics_history)]
        for title, metrics in grouped_metrics_data.items():
            for metric_name in metrics.keys():
                # Fill in values, using None for missing epochs
                metric_values = [d.get(metric_name, None) for d in self.metrics_history]
                grouped_metrics_data[title][metric_name] = metric_values
        
        # --- Plotting ---
        num_plots = len(grouped_metrics_data)
        fig, axes = plt.subplots(num_plots, 1, figsize=(12, 6 * num_plots), sharex=True)
        if num_plots == 1:
            axes = [axes]

        plot_idx = 0
        for title, metrics_data in grouped_metrics_data.items():
            ax = axes[plot_idx]
            for metric_name, values in metrics_data.items():
                # Filter out None values for continuous plotting
                valid_points = [(e, v) for e, v in zip(epochs, values) if v is not None]
                if valid_points:
                    e, v = zip(*valid_points)
                    ax.plot(e, v, marker='o', linestyle='-', label=metric_name)

            ax.set_ylabel(title, fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, linestyle='--', alpha=0.6)
            plot_idx += 1
            
        axes[-1].set_xlabel("Epoch", fontsize=12)
        fig.suptitle("Training Metrics Over Time", fontsize=16, y=0.99)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        plot_path = os.path.join(save_dir, "metrics_plot.png")
        try:
            fig.savefig(plot_path, dpi=120)
        except Exception as e:
            warnings.warn(f"Could not save metrics plot to {plot_path}. Error: {e}")
        finally:
            plt.close(fig)