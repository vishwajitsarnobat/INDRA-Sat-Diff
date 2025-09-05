# Example: Using the Climate Forecast Framework

This directory serves as a template for setting up **your own independent project** to run climate forecasting experiments. It demonstrates the standard workflow: installing the `climate-forecast` library and then using it in a separate project.

## Getting Started: A Two-Directory Workflow

It is highly recommended to keep your project separate from the cloned library code. This is a standard practice in software development that makes managing dependencies and your own code much easier.

You will have two main directories:

**1. The Library (cloned from GitHub)**
This contains the source code for the `climate-forecast` package. You'll install from here, but you won't work here.
```
/path/to/cloned/
└── indra_sat_diff/
    ├── climate_forecast/
    ├── examples/
    └── pyproject.toml
```

**2. Your Project (where you will work)**
This is where you will set up your experiment, manage your data, and run the pipeline.
```
/path/to/my_climate_project/
├── .venv/
├── data/
├── config.yaml
└── main.py
```

---

## Step 1: Setting Up Your Environment & Installing the Library

This part only needs to be done once.

**A. Clone the Library Repository**
If you haven't already, clone the `indra_sat_diff` repository to a convenient location on your machine.

```bash
git clone https://github.com/your-username/indra-sat-diff.git
# This creates a directory named 'indra_sat_diff'
```

**B. Create Your Project and Virtual Environment**
Navigate to where you want your project to live, create the project folder, and then create a virtual environment inside it.

```bash
# Go to your general workspace
cd ~/Workspace

# Create and enter your new project folder
mkdir my_climate_project
cd my_climate_project

# Create and activate a virtual environment using uv
uv venv
source .venv/bin/activate
```

**C. Install the `climate-forecast` Package**
Now, from inside your activated project environment, install the library from the cloned repository. Using the `-e` (editable) flag is crucial, as it allows you to make changes to the library's source code without needing to reinstall it.

```bash
# Make sure to replace the path with the actual path to where you cloned the repo
uv pip install -e /path/to/cloned/indra_sat_diff
```

Your environment is now set up! The `climate-forecast` package is available for you to import and use.

## Step 2: Creating Your Project from this Template

Now, let's set up your project using the files from the `examples` directory.

**A. Copy the Template Files**
Copy the `main.py` and `config.yaml` from the cloned repository's `examples` folder into your new project directory.

```bash
# Make sure you are inside 'my_climate_project'
# Replace the path with the actual path to the cloned repo
cp /path/to/cloned/indra_sat_diff/examples/{main.py,config.yaml} .
```

**B. Add Your Data**
Create a `data` directory and place your raw `.HDF5` data files inside it.

```bash
mkdir data
# Now, copy your raw .HDF5 files into the 'data/' directory
```

Your project is now fully set up and ready to run.

---

## Step 3: Running the Pipeline

All the following commands are run from inside your project directory (`my_climate_project/`).

**A. Configure Your Run**
Open `config.yaml`. It is pre-filled with recommended settings. You **must** review and confirm the settings in these sections:
-   `pipeline.output_dir`: Where all training results will be saved.
-   `preprocess`: The input directory (`data`), output directory, and geographic boundaries (`lat_range`, `lon_range`) for your data.
-   `data`: Ensure the variable names (`channels`, `latitude_variable_name`, etc.) match the structure of your HDF5 files.

**B. Preprocess the Data**
This step converts your raw data into a processed format ready for the model.

```bash
uv run main.py preprocess
```

This will create a `processed_data` directory (or as specified in `config.yaml`).

**C. Train the Model**
This runs the complete three-stage training pipeline. This process is computationally intensive and requires a GPU.

```bash
uv run main.py train
```

**D. Generate a Forecast**
After training, use the final model to generate a forecast. You must provide the path to the **last file** of an input sequence.

```bash
# Example: Replace with a real file from your processed_data directory
export LAST_INPUT_FILE="processed_data/your_last_input_file.HDF5"

uv run main.py forecast --input $LAST_INPUT_FILE
```
The script will save a static image, an animated GIF, and the raw forecast data to the `output/` directory by default.

---
## Customization

To run on hardware with limited VRAM, you can adjust the following parameters in `config.yaml`:

-   `optim.micro_batch_size`: Lower this to `1`.
-   `trainer.precision`: Set to `"16-mixed"` to save memory.
-   `model.vae.block_out_channels` and `model.latent_model.base_units`: Reduce these numbers to create a smaller model.