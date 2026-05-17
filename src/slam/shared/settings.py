import json
import sys
from pathlib import Path

# Safely resolve path relative to the repo's config directory
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.json"

try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        CFG = json.load(f)
except FileNotFoundError:
    print(f"\n[FATAL] Could not find configuration file at:\n  {CONFIG_PATH}")
    sys.exit(1)

# Centralized constants derived from config
DECIMATE_FACTOR = CFG["hardware"]["decimate_factor"]
ENABLE_INTEL_IGPU = CFG["hardware"]["enable_intel_igpu"]
DEPTH_SCALE = 1000.0
