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
  "equations_to_run": ["flow", "press", "vol"],

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
  }
}
```
The same content can be seen in config.json which you can modify before running the code