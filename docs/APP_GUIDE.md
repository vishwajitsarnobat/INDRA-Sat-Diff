# INDRA-Sat-Diff app — user guide

The app lets you look at satellite rainfall, make and verify nowcasts, and train models without editing
any configuration files. Everything runs on your own machine; the browser is only the display.

## 1. Start it

```bash
cd INDRA-Sat-Diff
.venv/bin/python -m app                  # then open http://127.0.0.1:8765
.venv/bin/python -m app --port 9000      # another port
.venv/bin/python -m app --host 0.0.0.0   # let colleagues on your network open http://<your-ip>:8765
```

Requirements: the project installed with `uv pip install -e ".[app]"`, and an NVIDIA GPU for forecasts and
training (viewing data works without one). Stop the server with Ctrl+C.

The top bar has the pages, the **dataset selector** (used by every page), a job indicator, the UTC clock,
the light/dark switch and `?` for help. Press **1–8** to switch pages, **L** to hide the side panel,
**T** to switch theme.

## 2. Look at data — *Viewer*

1. Pick a dataset in the top bar. The left panel shows its region, grid, period and gaps.
2. Use the time bar at the bottom: drag, click, or press **Space** (play/pause), **← →** (one step),
   **[ ]** (one hour), **Home/End**. Hatched marks on the bar are missing time steps.
3. Hover over the map for latitude, longitude and rain rate (mm/h). Zoom with the mouse wheel or **+ −**,
   **0** to fit the region.
4. *Layers*: rain field opacity, coastlines, India state boundaries, other borders, grid, region outline,
   smooth or cell display.
5. *Forecast from this time* opens Predict with the frame on screen as the last observation.

Frames load on demand and are cached, so even multi-year datasets open instantly.

## 3. Make a forecast — *Predict*

1. Choose a **model** (left panel). Its validated settings are filled in automatically.
2. Choose the **start time** (UTC). This is the *last observed* frame: the model uses it and the 6 frames
   before it (3.5 h) and forecasts the next 6 frames (+30 to +180 min). Keep *Only times with
   observations* ticked to be able to verify the forecast.
3. Options (defaults are the validated ones — change them only to experiment):
   - *Ensemble members*: independent samples (4 is the validated choice; more is slower).
   - *Sampler steps*: 20 is enough; more is slower and rarely better.
   - *Combine members into one forecast*: probability matching + intensities (validated for South
     India), probability threshold (validated for Northeast India), ensemble mean, or a single sample.
   - *Knowledge alignment* and its *target*: 1 × the mean input rain is the validated value.
   - *Precision*: fp16 is fastest; *Seed* makes a run repeatable.
4. Press **Run forecast**. A 4-member forecast takes about 20–40 s on a laptop GPU.

Reading the result:

| Row | Meaning |
|---|---|
| Input | the 7 observed frames used |
| Forecast | the combined forecast (or one member, chosen in *Member*) |
| Observed | IMERG truth, when available |
| Difference | forecast − observed (red: too much rain, blue: too little) |
| Probability | share of members with rain above the chosen threshold |
| Spread | disagreement between members (uncertainty) |

Below the maps: POD, FAR and CSI per lead time against persistence (“nothing moves”), at the selected
threshold. *Download arrays (.npz)* saves everything for your own analysis
(values: `mm/h = expm1(value · ln 101)`). Past forecasts stay listed on the left.

## 4. Replay an event — *Simulate*

Pick a period (for example a storm), how often to issue a forecast (every 1–3 h) and the threshold, then
**Run replay**. The app issues forecasts through the event and shows each next to what actually happened, with a
running skill chart against persistence. *Stop* ends it early.

## 5. Score a model — *Evaluate*

Choose a period and how often to start a forecast, then **Run evaluation**. Tick *Compare with and without
knowledge alignment* to score both. The result is a table and charts of POD, FAR, CSI and frequency bias per
lead time and threshold (*Download scores (.csv)*), saved under `artifacts/results/`. Use complete periods the model never trained on
(for the provided models: June–September 2022–2023).

## 6. Get new data — *Download*

1. Press *Draw box on map* and drag the region (or type it, or use the *South India* / *Northeast India* presets).
   A 12.8° × 12.8° box at 0.1° gives the 128 × 128 grid the models expect; other sizes work and are
   resampled. *Downscale factor* 2 or 4 makes a coarser, smaller grid.
2. Choose the product: IMERG **Final** (best quality, ~3.5 months delay), **Late** (~14 h) or **Early** (~4 h),
   and the period (UTC).
3. Give the dataset an identifier and press **Download and preprocess**. Only your box is downloaded from NASA (a few KB per frame), then
   preprocessed; the new dataset appears in the top bar.

You need a free NASA Earthdata account, with *NASA GESDISC DATA ARCHIVE* approved under
*Applications → Authorized Apps*, and this line in `~/.netrc` (then `chmod 600 ~/.netrc`):
```
machine urs.earthdata.nasa.gov login YOUR_USER password YOUR_PASSWORD
```

## 7. Train a model — *Train*

1. Pick the training dataset (top bar). Use one with several years of data.
2. **Model size.** Press **Measure GPU memory**: the app runs a few training steps at each compute level and
   recommends the largest model that fits your GPU (level 1 = full size, as trained at SAC on a 32 GB GPU;
   level 8 ≈ 1.4 GB; level 16 ≈ 0.9 GB).
3. **Schedule.** Epochs per stage, and the split: choose **by year** (for example train 2001–2020,
   validate 2022, test 2023) for any result you will report; a random split leaks between neighbouring
   time windows.
4. Name the model and press **Start training**. The three stages (VAE → alignment → diffusion) run one after the
   other; follow the live log in *Jobs*. The YAML the app will write is previewed live as you change settings.
5. The finished model appears in *Models* and in Predict.

To continue an interrupted run, open it in *Models* and press **Resume training**: finished stages are
reused and training continues from the last checkpoint, with memory settings that fit a 4 GB GPU.

## 8. Models and jobs

*Models* lists every model (region, grid, compute level, size, notes) with its training curves and its
validated forecast settings. *Jobs* lists everything the app ran — forecasts, evaluations, downloads,
training — with live logs and *Cancel*. Only one GPU job runs at a time; others wait in the queue.

## 9. Use your own data

Any rain (or other 2-D) product works if it is one HDF5/netCDF4 file per time step on a regular lat/lon
grid. Either download through the app, or put files in a folder and preprocess them:

```bash
climate-forecast preprocess --config my_region.yaml      # see climate_forecast/examples/
```

then describe the processed folder for the app in `artifacts/data/<name>/dataset.yaml`
(processed files in `artifacts/data/<name>/processed/`):

```yaml
name: My region (IMERG)
lat_range: [5.0, 17.8]
lon_range: [71.0, 83.8]
channel: precipitation        # variable name inside the files
time_variable: time
clip_value: 100.0             # mm/h used in normalisation
interval_minutes: 30
array_order: lon_lat          # IMERG stores [lon, lat]
time_source: file             # or "filename" if the embedded times are unreliable
```

## 10. Where things are

| What | Where |
|---|---|
| Models (weights, config, model card) | `artifacts/models/<name>/` |
| Datasets | `artifacts/data/<name>/` |
| Forecasts and logs of app jobs | `artifacts/jobs/<job id>/` |
| Evaluations | `artifacts/results/` |

## 11. Troubleshooting

| Problem | What to do |
|---|---|
| A job waits in the queue | another GPU job is running; see *Jobs* |
| “out of memory” | use a higher compute level, fewer ensemble members, or fp16 |
| Forecast is slow | the GPU may be overheating (laptops throttle); improve cooling, fewer steps |
| Download fails with 401/403 | check `~/.netrc` and the *Authorized Apps* approval; reset the Earthdata password if asked |
| “Need 7 consecutive frames” | pick a start time without gaps before it |
| Page looks stale after an update | reload with Ctrl+Shift+R |
