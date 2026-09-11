source /workspace/.venv/bin/activate
export DATA_ROOT=
export MODEL_ROOT=
export EVAL_ROOT=
export PROJECT_ROOT=
export HF_HOME=
export PIP_CACHE_DIR=

# The ArchesWeather repo and the era5-quantiles stats file are public, so
# downloads work anonymously and nothing is currently blocked by this. Fill it
# in if a gated repo is ever needed; leave it empty rather than setting a
# dummy string, which would make hf_hub send a bad Authorization header and
# turn anonymous 200s into 401s.
