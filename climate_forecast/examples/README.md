# Example configurations

Every command takes one YAML file that overrides `climate_forecast/configs/train.yaml`; set only what differs.

| File | Use |
|---|---|
| `config.yaml` | Annotated template: paths, region, data variables, model and training settings |
| `config_full.yaml` | Full-size model with a chronological (year) split, for a 24–32 GB GPU |
| `config_laptop.yaml` | `compute_level: 8` for a 4 GB GPU, same data layout |
| `regions/south_india_demo.yaml`, `regions/northeast_india_demo.yaml` | IMERG download + preprocessing for the two study regions (flood-event periods) |

Typical workflow (from the repository root, after `uv pip install -e .`):

```bash
climate-forecast download   --config climate_forecast/examples/regions/south_india_demo.yaml
climate-forecast preprocess --config climate_forecast/examples/regions/south_india_demo.yaml
climate-forecast fit-check  --config climate_forecast/examples/config_laptop.yaml
climate-forecast train      --config climate_forecast/examples/config_laptop.yaml
climate-forecast forecast   --config climate_forecast/examples/config_laptop.yaml --input <T0 file>
```

`--input` is the last observed frame (T0); the 7 frames ending there are the inputs. Forecast settings
(ensemble size, steps, post-processing, KA target) default to the `inference` section of the model's config;
override with `--ensemble`, `--steps`, `--postprocess`, `--ka-multiplier`, `--no-alignment`. All times are UTC.

The same steps are available without YAML in the web app: `.venv/bin/python -m app`.
