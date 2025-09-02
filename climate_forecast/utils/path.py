# climate_forecast/utils/path.py
import os

# Get the directory where the current file (path.py) is located.
# This makes the path relative to the package itself.
_utils_dir = os.path.dirname(os.path.abspath(__file__))

# Define a default directory for pretrained models at the root of the package.
# This will resolve to .../climate_forecast/models/
default_models_dir = os.path.abspath(os.path.join(_utils_dir, "..", "models"))
os.makedirs(default_models_dir, exist_ok=True)

# Specifically define the directory for pretrained metrics models, as required by lpips.py
default_pretrained_metrics_dir = os.path.join(default_models_dir, "pretrained_metrics")
os.makedirs(default_pretrained_metrics_dir, exist_ok=True)