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

import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import numpy as np
from sklearn.metrics import r2_score


import lib_residue_estimate

def get_breaths(df):
    breaths=[]
    breath_start=0
    for i in range(1,len(df)):
        if df.iloc[i-1]['phase'] == 0 and df.iloc[i]['phase'] == 1:
            #print(breath_start,i)
            breath_df=df.iloc[breath_start:i]
            
            breaths.append(breath_df)
            breath_start=i
    return breaths    


def get_max_signals_per_inspiration(arr,Phase):
    # Step 1: Find start and end indices of inspiration (1) segments
    inspiration_indices = np.where(Phase == 1)[0]
    splits = np.split(inspiration_indices, np.where(np.diff(inspiration_indices) != 1)[0] + 1)
    
    # Step 2: Get max of arr for each inspiration segment
    max_per_inspiration = [arr[segment].max() for segment in splits if len(segment) > 0]
    
    # Result
    #print("Max values per inspiration phase:", max_per_inspiration)    
    return max_per_inspiration





def estimate_general(
    equation_type,              # "flow", "press", or "vol"
    P0_status,                  # "shared" or "individual"
    lower_bounds,
    upper_bounds,
    x0,
    Flow, Press, Vol, Edi,
    breath_indices,
    Phase
):
    # Dynamic function selection from lib_residue_estimate
    if P0_status == 'shared':
        residual_fn = getattr(lib_residue_estimate, f"residuals_{equation_type[0]}_singleP0")
        estimate_fn = getattr(lib_residue_estimate, f"estimate{equation_type.capitalize()}_singleP0")
        args = (Flow, Press, Vol, Edi)
    else:
        residual_fn = getattr(lib_residue_estimate, f"residuals_{equation_type[0]}")
        estimate_fn = getattr(lib_residue_estimate, f"estimate{equation_type.capitalize()}")
        args = (Flow, Press, Vol, Edi, breath_indices)

    # Run optimization
    result = least_squares(
        residual_fn,
        x0,
        args=args,
        bounds=(lower_bounds, upper_bounds),
        verbose=0
    )

    # Compute estimated signal
    signal_est = estimate_fn(result.x, *args)

    # Extract common parameters
    R, E, NME = result.x[:3]
    Compliance = 1.0 / E if E != 0 else None
    Ppat = Edi * NME
    max_Ppat = float(np.median(np.array(get_max_signals_per_inspiration(Ppat, Phase))))

    # Determine ground truth and R²
    ground_truth = {'flow': Flow, 'press': Press, 'vol': Vol}[equation_type]
    r2 = r2_score(ground_truth, signal_est)

    if P0_status == 'shared':
        P0 = result.x[3]
        return {
            "R": float(R),
            "E": float(E),
            "NME": float(NME),
            "P0": float(P0),
            "Compliance": float(Compliance),
            "R2Score": float(r2),
            "max_Ppat": max_Ppat
        }
    elif P0_status=='individual':
        P0_values = result.x[3:]
        return {
            "R": float(R),
            "E": float(E),
            "NME": float(NME),
            "P0_values": P0_values.tolist(),
            "mean_P0_values": float(np.mean(P0_values)),
            "Compliance": float(Compliance),
            "R2Score": float(r2),
            "max_Ppat": max_Ppat
        }




def make_estimations(equation_to_run,P0_status,pid,support_level,files_directory,file_name,lower_bounds,upper_bounds,x0,numBreathsEachTime):
    file_path=f'{files_directory}/{file_name}'
    df=pd.read_csv(file_path)
    breaths=get_breaths(df)
    print(file_path,'number of breaths total=',len(breaths))
    result_all=[]
    for breath_start in range(0,len(breaths),numBreathsEachTime):
        #print(f'breath number {breath_start} to {breath_start+numBreathsEachTime}')
        breath_df=pd.concat(breaths[breath_start:breath_start+numBreathsEachTime])
        Edi=breath_df['Edi'].values      
        Vol=breath_df['Vol'].values        
        Press=breath_df['Press'].values
        Flow=breath_df['Flow'].values        
        Pes=breath_df['Pes'].values         
        dPes=breath_df['dPes'].values 
        Phase=breath_df['phase'].values     
        max_Pes_measured = np.median(np.array(get_max_signals_per_inspiration(-Pes, Phase)))

        # Build mapping of time point to breath index (0 to 14)
        breath_indices = []
        for b_idx, b in enumerate(breaths[breath_start:breath_start+numBreathsEachTime]):
            breath_indices.extend([b_idx] * len(b))
        breath_indices = np.array(breath_indices)

        result_dict = estimate_general(
            equation_type=equation_to_run,
            P0_status=P0_status,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            x0=x0,
            Flow=Flow,
            Press=Press,
            Vol=Vol,
            Edi=Edi,
            breath_indices=breath_indices,
            Phase=Phase
            )

        result_dict["max_Pes"]=float(max_Pes_measured)

        #rint(result_dict)
        result_all.append(result_dict)


        
        #process_params(equation_to_run,P0_status,Flow, Press, Vol, Edi,breath_indices, result, Pes )


    # return a list of dicts
    return result_all


def generate_patient_plots(
    pid,
    base_path,
    equation_to_run,
    plots_to_generate,
    P0_status,
    support_levels=[1, 2, 3],
    colors=['r', 'g', 'b']
):
    # --- SCATTER PLOT ---
    if 'scatter' in plots_to_generate:
        plt.figure()
        min_val, max_val = 0, 0
        for support_level in support_levels:
            result_file_path = f'{base_path}/{equation_to_run}/pat{pid}/supp_level_{support_level}_res.csv'
            if not os.path.exists(result_file_path):
                print(f" Scatterplot Skipping missing file: {result_file_path}")
                continue

            df = pd.read_csv(result_file_path)
            x, y = df['max_Pes'].values, df['max_Ppat'].values
            plt.scatter(x, y, label=f'Support {support_level}')
            min_val = min(min(x), min(y), min_val)
            max_val = max(max(x), max(y), max_val)

        plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='gray', linewidth=1)
        plt.xlabel('Pes')
        plt.ylabel('Ppat')
        plt.grid(True)
        plt.title('Scatter plot Pes vs Ppat')
        plt.legend()
        plt.savefig(f'{base_path}/{equation_to_run}/pat{pid}/scatter_Pes_vs_Ppat.pdf')
        plt.close()

    # --- BLAND–ALTMAN PLOT ---
    if 'bland-altman' in plots_to_generate:
        plt.figure(figsize=(10, 5))
        all_means, all_diffs = [], []

        for support_level in support_levels:
            result_file_path = f'{base_path}/{equation_to_run}/pat{pid}/supp_level_{support_level}_res.csv'
            if not os.path.exists(result_file_path):
                print(f" Bland-altman Skipping missing file: {result_file_path}")
                continue

            df = pd.read_csv(result_file_path)
            Ppat, Pes = df['max_Ppat'].values, df['max_Pes'].values
            means = (Ppat + Pes) / 2
            diffs = Ppat - Pes
            all_means.extend(means)
            all_diffs.extend(diffs)
            plt.scatter(means, diffs, s=12, alpha=0.6, label=f'Support {support_level}')

        if all_diffs:
            all_means = np.array(all_means)
            all_diffs = np.array(all_diffs)
            bias = np.mean(all_diffs)
            sd = np.std(all_diffs)
            upper, lower = bias + 1.96 * sd, bias - 1.96 * sd

            plt.axhline(bias, linestyle='--', color='gray', linewidth=1, label=f'Bias = {bias:.2f}')
            plt.axhline(upper, linestyle=':', color='blue', linewidth=1, label=f'+1.96 SD = {upper:.2f}')
            plt.axhline(lower, linestyle=':', color='blue', linewidth=1, label=f'-1.96 SD = {lower:.2f}')

            plt.text(
                0.05 * (plt.xlim()[1] - plt.xlim()[0]) + plt.xlim()[0],
                0.95 * (plt.ylim()[1] - plt.ylim()[0]) + plt.ylim()[0],
                f'Bias = {bias:.2f}\nSD = {sd:.2f}',
                fontsize=9, verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray')
            )

        plt.title(f'Bland–Altman Plot: Ppat vs Pes (Patient {pid})')
        plt.xlabel('Mean of Ppat and Pes')
        plt.ylabel('Ppat - Pes')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{base_path}/{equation_to_run}/pat{pid}/bland_altman_combined.pdf')
        plt.close()

    # --- LINE PLOTS ---
    if 'line' in plots_to_generate:
        chosen_column_list = (
            ['E', 'R', 'NME', 'mean_P0_values', 'Compliance'] if P0_status == 'individual'
            else ['E', 'R', 'NME', 'P0', 'Compliance']
        )

        for column in chosen_column_list:
            plt.figure(figsize=(10, 4))
            for i, support_level in enumerate(support_levels):
                result_file_path = f'{base_path}/{equation_to_run}/pat{pid}/supp_level_{support_level}_res.csv'
                if not os.path.exists(result_file_path):
                    print(f"    Skipping missing file: {result_file_path}")
                    continue

                df = pd.read_csv(result_file_path)
                if column not in df.columns:
                    print(f"    Skipping: {column} not in {result_file_path}")
                    continue

                y_vals = df[column].values
                x_vals = np.arange(len(y_vals))
                plt.plot(x_vals, y_vals, color=colors[i], label=f'Support {support_level}')

            plt.title(f'{column} across support levels (Patient {pid})')
            plt.xlabel('Breath Window Index')
            plt.ylabel(column)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'{base_path}/{equation_to_run}/pat{pid}/line_{column}.pdf')
            plt.close()