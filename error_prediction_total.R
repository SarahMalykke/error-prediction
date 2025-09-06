##### ERROR PREDICTION TOTAL (TrialsSinceError == trial)#####

library(dplyr)
library(lme4)
library(ggplot2)

library(dplyr)

# Define the output directory
output_dir <- "/Users/SarahMalykke/Documents/GW/AirportScanner/Manuscript1"
predictions_list_total <- list()

# Define trial ranges
pre_error_range <- seq(-15, -1)
post_error_range <- seq(0, 15)
trials_since_error_values <- sort(unique(clean_combined_data$TrialsSinceError)) # pre and post

# Get unique users
total_users <- unique(clean_combined_data$UserId)
cat("Total users:", length(total_users), "\n")

# Create an empty list to store data
trial_data_list_total <- list()

# First define trial loop
for (trial in trials_since_error_values) {
  cat("Processing TrialsSinceError Trial:", trial, "\n")
  
  # Filter training data up to the current trial
  trial_data <- clean_combined_data %>% filter(TrialsSinceError == trial)
  
  cat("Number of rows in trial_data for Trial", trial, ":", nrow(trial_data), "\n")
  
  # Fit LME model on training data
  lme_model_total <- tryCatch(
    {
      lmer(Total_RT_log ~ 1 + (1 | TrialNumber) + (1 | ItemId), data = trial_data)
    },
    error = function(e) {
      cat("Error fitting model for Trial:", trial, "-", e$message, "\n")
      NULL
    }
  )
  
  if (is.null(lme_model_total)) next
  
  # Extract residuals
  trial_data <- trial_data %>%
    mutate(Residuals = resid(lme_model_total))
  
  # Save trial data with residuals
  output_filename <- file.path(output_dir, paste0("trial_data_total_residuals_", trial, ".csv"))
  write.csv(trial_data, file = output_filename, row.names = FALSE)
  
  # Store in a list for use within the R session
  trial_data_list_total[[as.character(trial)]] <- trial_data
}

# Load necessary libraries
library(tidyverse)

# Initialize a list to store transformed data for each trial
wide_data_list_total <- list()

# Loop through each trial's dataset
for (trial in names(trial_data_list_total)) {
  cat("Processing Trial:", trial, "\n")
  
  # Extract trial data
  trial_data <- trial_data_list_total[[trial]]
  
  # Ensure necessary columns exist
  if (!all(c("UserId", "TrialsSinceError", "Residuals", "Group") %in% colnames(trial_data))) {
    cat("Skipping trial", trial, "due to missing columns.\n")
    next
  }
  
  # Add a new column `Group_numeric` (Error = 1, NoError = 0)
  trial_data <- trial_data %>%
    mutate(Group_numeric = ifelse(Group == "Error", 1, 0))
  
  # Transform to wide format
  wide_data_total <- trial_data %>%
    select(UserId, TrialsSinceError, Residuals, Group, Group_numeric) %>%
    pivot_wider(names_from = TrialsSinceError, values_from = Residuals, names_prefix = "Residual_")
  
  # Store the transformed dataset
  wide_data_list_total[[trial]] <- wide_data_total
  
  # Save as CSV
  output_filename <- file.path(output_dir, paste0("wide_trial_data_total_residuals_", trial, ".csv"))
  write.csv(wide_data_total, file = output_filename, row.names = FALSE)
}


# Load necessary libraries
library(tidyverse)
library(caret)

# Initialize a list to store accuracy results
accuracy_results_total <- list()

# Initialize a list to store predictor statistics (mean & SD for pred_prob ≈ 0.5)
dt_predictor_stats_results_total <- list()


# Loop through each trial's dataset in wide format
for (trial in names(wide_data_list_total)) {
  cat("\nProcessing LOO-CV for Trial:", trial, "\n")
  
  # Extract wide-format trial data
  trial_data <- wide_data_list_total[[trial]]
  
  # Ensure trial is a numeric value, then convert to character
  trial_str <- as.character(trial)  
  
  # Concatenate "Residual_" with the cleaned trial string
  latest_residual_col <- paste0("Residual_", trial_str)
  
  # Ensure the residual column exists
  if (!(latest_residual_col %in% colnames(trial_data))) {
    cat("Skipping trial", trial, "- No residual column found for", latest_residual_col, "\n")
    next
  }
  
  # Ensure Group_numeric exists in the dataset
  if (!"Group_numeric" %in% colnames(trial_data)) {
    cat("Skipping trial", trial, "- Missing 'Group_numeric' column.\n")
    next
  }
  
  # Skip if dataset has only one class in Group_numeric
  if (length(unique(trial_data$Group_numeric)) < 2) {
    cat("Skipping trial", trial, "due to insufficient class variance.\n")
    next
  }
  
  # Handle NAs: Drop rows with NA in the latest residual column
  cat("Number of rows before removing NAs:", nrow(trial_data), "\n")
  trial_data <- trial_data %>% drop_na(latest_residual_col)
  cat("Number of rows after removing NAs:", nrow(trial_data), "\n")
  
  # **New Check: Skip trial if fewer than 3 rows remain**
  if (nrow(trial_data) < 3) {  
    cat("Skipping trial", trial, "- Not enough data for LOO-CV.\n")
    next
  }
  
  # Initialize vectors to store predictions and predicted probabilities for the current trial
  predictions <- rep(NA, nrow(trial_data))
  pred_probs   <- rep(NA, nrow(trial_data))
  
  # Perform Leave-One-Out Cross-Validation (LOO-CV) for the current trial
  for (i in 1:nrow(trial_data)) {
    # Define training set (exclude i-th row)
    train_data <- trial_data[-i, ]
    
    # Define test set (i-th row)
    test_data <- trial_data[i, , drop = FALSE]
    
    # Train logistic regression model using only the current residual column
    formula <- as.formula(paste0("Group_numeric ~ `", latest_residual_col, "`"))
    model <- glm(formula, data = train_data, family = binomial)
    
    # Predict probability for test case
    pred_prob <- predict(model, newdata = test_data, type = "response")
    pred_probs[i] <- pred_prob
    
    # Convert probability to class label (0.5 threshold)
    predictions[i] <- ifelse(pred_prob > 0.5, 1, 0)
  }
  
  # **Now, after the inner loop, create the trial-level predictions data frame:**
  trial_predictions <- data.frame(
    UserId    = trial_data$UserId,
    pred_prob = pred_probs,
    Group     = trial_data$Group,
    Trial     = as.numeric(trial)
  )
  
  # Append the predictions for this trial to your list
  predictions_list_total[[trial]] <- trial_predictions
  
  # Compute accuracy
  actual <- trial_data$Group_numeric
  accuracy <- mean(predictions == actual, na.rm = TRUE)
  # Compute standard deviation (SD) and standard error (SE)
  sd_predictions <- sd(predictions, na.rm = TRUE)
  se_predictions <- sd_predictions / sqrt(length(na.omit(predictions)))  # SE = SD / sqrt(n)
  
  # Store results for this trial
  accuracy_results_total[[trial]] <- list(
    Accuracy = accuracy,
    SD = sd_predictions,
    SE = se_predictions
  )
  
  cat("Trial", trial, "LOO-CV Accuracy:", round(accuracy, 3), "SD:", round(sd_predictions, 3), "SE:", round(se_predictions, 3), "\n")
  
  ### **Compute Mean & SD of Predictors for pred_prob ≈ 0.5** ###
  threshold_diff <- abs(pred_probs - 0.5)
  threshold_indices <- which(threshold_diff <= 0.01)  # Find cases where pred_prob ≈ 0.5
  
  if (length(threshold_indices) > 0) {
    relevant_data <- trial_data[threshold_indices, latest_residual_col, drop = FALSE]  # Select predictor values
    
    predictor_mean <- mean(relevant_data[[latest_residual_col]], na.rm = TRUE)  # Compute mean
    predictor_sd <- sd(relevant_data[[latest_residual_col]], na.rm = TRUE)  # Compute SD
    
    # Store in results list
    dt_predictor_stats_results_total[[trial]] <- data.frame(
      TrialsSinceError = trial,
      Predictor = latest_residual_col,
      Mean = predictor_mean,
      SD = predictor_sd
    )
    
    print(dt_predictor_stats_results_total[[trial]])  # Print the data frame for this trial
    
  } else {
    cat("Skipping Mean/SD Calculation for TrialsSinceError", trial, "- No pred_prob ≈ 0.5 cases found.\n")
  }
}

# Convert accuracy results into a dataframe
accuracy_df_total <- do.call(rbind, lapply(names(accuracy_results_total), function(trial) {
  data.frame(
    TrialsSinceError = as.numeric(trial),
    Accuracy = accuracy_results_total[[trial]]$Accuracy,
    SD = accuracy_results_total[[trial]]$SD,
    SE = accuracy_results_total[[trial]]$SE
  )
}))

# Convert predictor statistics to a single dataframe
dt_predictor_stats_df_total <- bind_rows(dt_predictor_stats_results_total)

# Save accuracy results
write.csv(accuracy_df_total, file.path(output_dir, "accuracy_results_total.csv"), row.names = FALSE)

# Save predictor statistics
write.csv(dt_predictor_stats_df_total, file.path(output_dir, "pred_stats_total.csv"), row.names = FALSE)

# View results
View(accuracy_df_total)
View(dt_predictor_stats_df_total)




