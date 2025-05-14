# 01SingleEquationMethodv1

## Data
Can be found in data/study1
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

The equations being fitted are

$$
\text{Flow}(t) = \frac{\text{Paw}(t)}{R} - \frac{\text{Volume}(t) \cdot E}{R} + \frac{\text{Edi}(t) \cdot \text{NME}}{R} - \frac{P_0(b_n)}{R}
$$

