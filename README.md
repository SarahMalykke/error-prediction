# Predicting Errors from Trial-Level Residuals

## Overview
Predicts whether a trial belongs to the **Error** or **NoError** group using **trial-specific residualized log RTs** and **leave-one-out cross-validated** logistic regression.

## Directory Structure

```error-prediction/
├── error_prediction_total.R            # Residual log RT per trial relative to error; LOO-CV logistic regression
└── README.md
```

### `error_prediction_total.R`

**Purpose:**  
Predict whether a trial belongs to the **Error** or **NoError** group using trial-specific residual log RTs. For each `TrialsSinceError`, the script extracts residuals via intercept-only LME, pivots to wide format, and runs leave-one-out cross-validated logistic regression using the residual at the current trial index as the sole predictor.

**Input:**  
- `clean_combined_data.csv` — combined, matched dataset with at least:  
  `UserId`, `Group` ∈ {`Error`, `NoError`}, `TrialNumber`, `ItemId`, `Total_RT_log`, `TrialsSinceError`.  

**Steps:**  
- **Trial loop:** For each `TrialsSinceError` in the data.  
- **Residuals:** Fit intercept-only LME on `Total_RT_log` with random intercepts for `TrialNumber` and `ItemId`:  
  `Total_RT_log ~ 1 + (1|TrialNumber) + (1|ItemId)`  
  Extract per-trial residuals for the current trial.  
- **Wide transform:** Build a wide row per user with columns for each trial relative to error.  
- **Target encoding:** Add `Group_numeric` (Error = 1, NoError = 0).  
- **LOO-CV logistic regression:** Using only `Residual_` as the predictor, run LOO-CV to classify Error vs NoError; collect predicted probability (`pred_prob`) and class for each held-out user.  
- **Accuracy & uncertainty:** Compute accuracy, SD, and SE at trial each trial relative to error.  
- **Near-threshold probe:** For cases with `|pred_prob − 0.5| ≤ 0.01`, compute the predictor’s mean and SD to characterize decision boundaries.  
- **Aggregate:** Bind accuracy results and near-threshold stats across all trials relative to error.

**Output:**  
- `accuracy_results_total.csv` — trialwise accuracy with SD and SE.  
- `pred_stats_total.csv` — mean/SD of `Residual_` where `pred_prob ≈ 0.5`.  

---

## Contact Information

For any questions regarding this project, please contact:

**Name:** Sarah Malykke  
**Email:** sarahmalykke@gwu.edu 
