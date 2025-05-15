import json
import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

from scipy.optimize import least_squares
import numpy as np
import random


import lib


colors = ['r', 'g', 'b']  # Support levels 1, 2, 3


# --- Load config ---
with open('config.json', 'r') as f:
    config = json.load(f)


# --- Parse fields from config ---
files_directory = config["files_directory"]
equation_to_run = config["equation_to_run"]
limits = config["limits"]
numBreathsEachTime = config["breath_config"]["numBreathsEachTime"]
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

for pid in patient_ids:
    dir_path = os.path.join(base_path, equation_to_run, f"pat{pid}")
    os.makedirs(dir_path, exist_ok=True)

# --- Create consolidated plot directory (optional) ---
if consolidate_plots:
    plot_path = os.path.join(base_path, "consolidated_plots")
    os.makedirs(plot_path, exist_ok=True)    
    plot_path = os.path.join(base_path, equation_to_run, "consolidated_plots")
    os.makedirs(plot_path, exist_ok=True)

print("✅ Directories created:")
print(f"Base path: {base_path}")



# Core parameters
R_min = limits["R"]["min"]
R_start = limits["R"]["start"]
R_max = limits["R"]["max"]

E_min = limits["E"]["min"]
E_start = limits["E"]["start"]
E_max = limits["E"]["max"]

NME_min = limits["NME"]["min"]
NME_start = limits["NME"]["start"]
NME_max = limits["NME"]["max"]

P0_min = limits["P0"]["min"]
P0_start = limits["P0"]["start"]
P0_max = limits["P0"]["max"]

# Determine P0 structure
if P0_status == "shared":
    x0 = [R_start, E_start, NME_start, P0_start]
    lower_bounds = [R_min, E_min, NME_min, P0_min]
    upper_bounds = [R_max, E_max, NME_max, P0_max]
elif P0_status == "individual":
    x0 = [R_start, E_start, NME_start] + [P0_start] * numBreathsEachTime
    lower_bounds = [R_min, E_min, NME_min] + [P0_min] * numBreathsEachTime
    upper_bounds = [R_max, E_max, NME_max] + [P0_max] * numBreathsEachTime
else:
    raise ValueError("P0_status must be 'shared' or 'individual'")

print('Range of R, E, NME and P0')
for i in range(len(lower_bounds)):
    print(lower_bounds[i],upper_bounds[i])

print('starting value',x0)


print(equation_to_run)
for pid in patient_ids:
    print(f'Patient:{pid}')
    for support_level in range(1,4):
        print(f'support_level:{support_level}')
        # create filename
        file_name=f'meas_pat{pid}_cfg1_idx{support_level}.csv'            
        if not os.path.exists(f'{files_directory}/{file_name}'):
            print(f'{files_directory}/{file_name}', 'missing')
            continue                                    
        result_pid_support_level=lib.make_estimations(equation_to_run,P0_status,pid,support_level,files_directory,file_name,lower_bounds,upper_bounds,x0,numBreathsEachTime)
        result_pid_support_level_df=pd.DataFrame(result_pid_support_level)
        result_pid_support_level_df.to_csv(f'{base_path}/{equation_to_run}/pat{pid}/supp_level_{support_level}_res.csv',index=False)
        
        # now to generate plots

    lib.generate_patient_plots(
        pid=pid,
        base_path=base_path,
        equation_to_run=equation_to_run,
        plots_to_generate=plots_to_generate,
        P0_status=P0_status
        )
'''
    if 'scatter' in plots_to_generate:    
        plt.figure()
        min_val=0
        max_val=0
        for support_level in range(1,4):
            
            result_file_path = f'{base_path}/{equation_to_run}/pat{pid}/supp_level_{support_level}_res.csv'
            if not os.path.exists(result_file_path):
                print(f" Scatterplot Skipping missing file: {result_file_path}")
                continue

            result_pid_support_level_df=pd.read_csv(result_file_path)

            x = result_pid_support_level_df['max_Pes']
            y = result_pid_support_level_df['max_Ppat']

            
            plt.scatter(x, y,label=f'ps{support_level}')
            

            min_val = min(min(x), min(y),min_val)
            max_val = max(max(x), max(y),max_val)
        plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='gray', linewidth=1)            
        plt.xlabel('Pes')
        plt.ylabel('Ppat')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{base_path}/{equation_to_run}/pat{pid}/scatter_Pes_vs_Ppat.pdf')        
        #plt.savefig(f'{base_path}/{equation_to_run}/pat{pid}/scatter_supp_level_{support_level}.png')
        #plt.savefig(f'{base_path}/{equation_to_run}/pat{pid}/scatter_supp_level_{support_level}.svg')
        plt.close()  # <-- Prevents window from popping up or lingering in memory

    if 'bland-altman' in plots_to_generate:
        plt.figure(figsize=(10, 5))

        all_means = []
        all_diffs = []

        for support_level in range(1, 4):
            result_file_path = f'{base_path}/{equation_to_run}/pat{pid}/supp_level_{support_level}_res.csv'
            if not os.path.exists(result_file_path):
                print(f" Bland-altman Skipping missing file: {result_file_path}")
                continue

            result_df = pd.read_csv(result_file_path)
            Ppat = result_df['max_Ppat'].values
            Pes = result_df['max_Pes'].values
            mean_vals = (Ppat + Pes) / 2
            diff_vals = Ppat - Pes

            # Accumulate for global bias
            all_means.extend(mean_vals)
            all_diffs.extend(diff_vals)

            # Plot support-level points
            plt.scatter(mean_vals, diff_vals, s=12, alpha=0.6, label=f'Support {support_level}')

        if all_diffs:
            all_means = np.array(all_means)
            all_diffs = np.array(all_diffs)

            # Shared/global bias and limits
            bias = np.mean(all_diffs)
            sd = np.std(all_diffs)
            upper = bias + 1.96 * sd
            lower = bias - 1.96 * sd

            # Add shared bias lines
            plt.axhline(bias, linestyle='--', color='gray', linewidth=1, label=f'Bias = {bias:.2f}')
            plt.axhline(upper, linestyle=':', color='blue', linewidth=1, label=f'+1.96 SD = {upper:.2f}')
            plt.axhline(lower, linestyle=':', color='blue', linewidth=1, label=f'-1.96 SD = {lower:.2f}')

            # Add bias text box
            plt.text(0.05 * (plt.xlim()[1] - plt.xlim()[0]) + plt.xlim()[0],
                     0.95 * (plt.ylim()[1] - plt.ylim()[0]) + plt.ylim()[0],
                     f'Bias = {bias:.2f}\nSD = {sd:.2f}',
                     fontsize=9, verticalalignment='top',
                     bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray'))

        # Decorations
        plt.title(f'Bland–Altman Plot: Ppat vs Pes (Patient {pid})')
        plt.xlabel('Mean of Ppat and Pes')
        plt.ylabel('Ppat - Pes')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Save
        save_path = f'{base_path}/{equation_to_run}/pat{pid}/bland_altman_combined'
        plt.savefig(f'{save_path}.pdf')
        plt.close()


    if 'line' in plots_to_generate:
        
        if P0_status=='individual':
            chosen_column_list = ['E', 'R', 'NME', 'mean_P0_values','Compliance']
        elif P0_status=='shared':
            chosen_column_list = ['E', 'R', 'NME', 'P0','Compliance']

        for chosen_column in chosen_column_list:
            print(f'Generating line plot for: {chosen_column}')
            plt.figure(figsize=(10, 4))

            for support_level in range(1, 4):
                print(f'  Support level: {support_level}')
                result_file_path = f'{base_path}/{equation_to_run}/pat{pid}/supp_level_{support_level}_res.csv'
                
                if not os.path.exists(result_file_path):
                    print(f"    Skipping missing file: {result_file_path}")
                    continue

                result_df = pd.read_csv(result_file_path)

                if chosen_column not in result_df.columns:
                    print(f"    Skipping: {chosen_column} not in {result_file_path}")
                    continue

                y_vals = result_df[chosen_column].values
                x_vals = range(len(y_vals))
                plt.plot(x_vals, y_vals, color=colors[support_level - 1], label=f'Support {support_level}')

            plt.title(f'{chosen_column} across support levels (Patient {pid})')
            plt.xlabel('Breath Window Index')
            plt.ylabel(chosen_column)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            save_path = f'{base_path}/{equation_to_run}/pat{pid}/line_{chosen_column}.pdf'
            plt.savefig(save_path)
            plt.close()
'''
