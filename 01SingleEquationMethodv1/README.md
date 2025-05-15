# 01SingleEquationMethodv1

## Data
Can be found in the top level directory data/study1v1
meas_pat16_cfg1_idx2.csv might be missing as it is too bog to be pushed through git
Consists of csv files for 26 patients and 3 support levels each.
For patient 12 only 2 support levels
CSV files are geerated from MATLAB.

Columns of interest are
- Edi
- Vol
- Press
- Flow
- Pes
- phase

Note that the naming convention of the files are a braeking point.
We expect the files to be named as follows
- meas_pat1_cfg1_idx1.csv
- meas_pat1_cfg1_idx2.csv
- meas_pat1_cfg1_idx3.csv
...
- meas_pat26_cfg1_idx1.csv
- meas_pat26_cfg1_idx2.csv
- meas_pat26_cfg1_idx3.csv

We assume "cfg1" to be fixed and idx to range from 1 to 3. While reading the files, we shall check if the file exists and skip otherwise.

## Equations

The equations being fitted are

$$
\text{Flow}(t) = \frac{\text{Paw}(t)}{R} - \frac{\text{Volume}(t) \cdot E}{R} + \frac{\text{Edi}(t) \cdot \text{NME}}{R} - \frac{P_0(b_n)}{R}
$$


$$
\text{Paw}(t) = \text{Flow}(t) \cdot R + \text{Volume}(t) \cdot E - \text{Edi}(t) \cdot \text{NME} + P_0(b_n)
$$


$$
\text{Volume}(t) = \frac{\text{Paw}(t)}{E} - \frac{\text{Flow}(t) \cdot R}{E} + \frac{\text{Edi}(t) \cdot \text{NME}}{E} - \frac{P_0(b_n)}{E}
$$

We are trying to estimate the parameters E, R, NME and P0

## Parameter Configuration

### Range of estimated parameters

The main variability that we have considered here is, the P0 values - whether we want to estimate them per breath or per k breaths.
Some of the configuration parameters used are
- numBreathsEachTime: How many breaths to be processed at each iteration, default 15
- Limit of E 
	- E_min=0.033, E_start=0.1, E_max=0.2
- Limit of R
	- R_min=0.0001, R_start=5, R_max=100
- Limit of NME
	- NME_min=0, NME_start=2, NME_max=5
- Limit of P0
	- P0_min=-10, P0_start=0, P0_max=10

### How to set parameters

When you run the code, you make the following choices:
- Which equation you want to run the code for ['flow', 'press', 'vol'] - a list where you keep 1 to 3 values
- What your limits and starting values will be (as shown in the above list)
- Do you want to generate P0 for each breath or each 15 breaths, defined as P0_status which can be either 'individual' or 'shared'

All these choices should be made in the local config.json file.
Example below
```json
{
  "files_directory": "../data/study1v1",
  "equation_to_run": "flow", // "press", "vol"

  "limits": {
    "E": {
      "min": 0.033,
      "start": 0.1,
      "max": 0.2
    },
    "R": {
      "min": 0.0001,
      "start": 5.0,
      "max": 100.0
    },
    "NME": {
      "min": 0.0,
      "start": 2.0,
      "max": 5.0
    },
    "P0": {
      "min": -10.0,
      "start": 0.0,
      "max": 10.0
    }
  },

  "breath_config": {
    "numBreathsEachTime": 15,
    "P0_status": "individual"  // Options: "individual" or "shared"
  },
  "result_directory":"results",
  "patients": "3:7",
  "plots":["scatter", "bland-altman", "line"],
  "consolidate_plots": "True", // False
  "title": "Basic Test"
}
```
The same content can be seen in config.json which you can modify before running the code

### Results

The results will be saved in the result directory according to equations chosen and P0_status
For example, if result_directory is 'results', equations_to_run contains ["flow", "press"], P0_status is "individual" and patients is 5:7, results will be stored in the following directory
- results/\<title\>_\<timestamp\>/individual/flow/pat5/
- results/\<title\>_\<timestamp\>/individual/flow/pat6/
- results/\<title\>_\<timestamp\>/individual/flow/pat7/
- results/\<title\>_\<timestamp\>/individual/press/pat5/
- results/\<title\>_\<timestamp\>/individual/press/pat6/
- results/\<title\>_\<timestamp\>/individual/press/pat7/
Timestamp is useful to denote the experiment date. The location can be generalized as follows

- results/\<title\>_\<timestamp\>/\<P0_status\>/\<equation_to_run\>/pat\<patient_number\>/

They will be csv files containing columns as follows:
- breath_start
- breath_end
- R2Score
- E
- R
- NME
- P0 (single value if P0_status is shared, list of values of P0_status is individual)
- mean_P0 (when we are using individual P0 values)
- Compliance
- Ppat
- dPes

#### Plots

You will notice two config variables plots and consolidate_plots
plots is a list containing the plots that should be prepared for all patients and support levels
- "scatter" - scatter plot of Ppat against Pes for each patient and support level
- "bland-altman" - Bland altman plot comparing Ppat and Pes
- "line" - line plots for 'E', 'R', 'NME', 'Compliance' and 'mean_P0'/'P0'
These plots will be stored in the same directory where the csv files will be created as seen above

The boolean consolidate_plots asks whether user wants to store plots that will consolidate for all patients
These plots will be stored in 
results/\<title\>_\<timestamp\>/\<P0_status\>/\<equations_to_run\>/consolidated_plots/ 

	and 

results/\<title\>_\<timestamp\>/\<P0_status\>/consolidated_plots/