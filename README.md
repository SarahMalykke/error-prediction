# error-prediction
Trial-level error prediction using residualized log RTs. For each TrialsSinceError index, fit intercept-only LMEs to extract residuals, and run LOO-CV logistic regression to classify Error vs NoError. Outputs accuracy (±SE), trialwise predictions, and near-threshold predictor stats.
