#!/usr/bin/env bash
# Overnight validation cache: tune on 2022, test on 2023, both regions, KA off / 1x / 2x.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
$PY tools/gen_validation.py south_jul2025 sac_valid_south --year 2022 --every 8
$PY tools/gen_validation.py ne_jul2025    sac_valid_ne    --year 2022 --every 8
$PY tools/gen_validation.py south_jul2025 sac_valid_south --year 2023 --every 6
$PY tools/gen_validation.py ne_jul2025    sac_valid_ne    --year 2023 --every 6 --variants noka ka2
echo ALL_DONE
