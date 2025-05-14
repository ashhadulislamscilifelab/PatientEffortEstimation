import json
import os
from datetime import datetime

# --- Load config ---
with open('config.json', 'r') as f:
    config = json.load(f)

# --- Parse fields from config ---
equations_to_run = config["equations_to_run"]
limits = config["limits"]
num_breaths = config["breath_config"]["numBreathsEachTime"]
P0_status = config["breath_config"]["P0_status"]
result_directory = config["result_directory"]
plots_to_generate = config["plots"]
consolidate_plots = config["consolidate_plots"] == "True"
title = config["title"]

# --- Parse patient range like "3:7" -> [3, 4, 5, 6, 7] ---
start_pat, end_pat = map(int, config["patients"].split(":"))
patient_ids = list(range(start_pat, end_pat + 1))

# --- Create base result path ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_path = os.path.join(result_directory, f"{title}_{timestamp}", P0_status)

# --- Create directories per patient and equation ---
for eq in equations_to_run:
    for pid in patient_ids:
        dir_path = os.path.join(base_path, eq, f"pat{pid}")
        os.makedirs(dir_path, exist_ok=True)

# --- Create consolidated plot directory (optional) ---
if consolidate_plots:
    for eq in equations_to_run:
        plot_path = os.path.join(base_path, eq, "consolidated_plots")
        os.makedirs(plot_path, exist_ok=True)

print("✅ Directories created:")
print(f"Base path: {base_path}")