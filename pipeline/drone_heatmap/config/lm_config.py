import os
import json
from pathlib import Path

# Locate the json file right alongside this script file
CONFIG_JSON_PATH = Path(__file__).parent / "lm_config.json"

# Default fallback fallback block if json is missing
if not CONFIG_JSON_PATH.exists():
    raise FileNotFoundError(f"Missing base configuration file layout at {CONFIG_JSON_PATH}")

with open(CONFIG_JSON_PATH, "r") as f:
    _raw_config = json.load(f)

# Expose fields as standard python modules variables
DATASET_ROOT = Path(_raw_config.get("DATASET_ROOT", "/Users/neal/Downloads/Train/Train"))
VLM_CONFIG = _raw_config.get("VLM_CONFIG", {})
LLM_CONFIG = _raw_config.get("LLM_CONFIG", {})