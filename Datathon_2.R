# Welcome to my Datathon Project Submission, by James Pavlicek.
# Project Title: Predicting Texas Public Universities Retention Rate With Multi Variable Linear Regression and Neural Networks in R
# This Project and it's supporting files were built from Feb 15th to Feb 16th for use in Texas State University's Datathon Competition.
# The code has received an additional revision on Feb 19th with comments to provide easier reader interpretation.
# This code is free use and anyone can use it.
# This code has two parts. You are currently reading "Datathon_2.R" which contains the code for the Neural Networks.
# The other part is "Datathon.R" which will utilize Multi Variable Linear Regression for a similar analysis.
# The last document you will need is "TXEDU17_23" which can be found at my github: https://github.com/james-pavlicek
# I hope you enjoy and reach out me if you have any questions at all! (https://www.jamespavlicek.com/) (https://www.linkedin.com/in/jamespavlicek/) 

# James Pavlicek / 2/16/2024 / Datathon / Neural Networks

#Code is partial written with the assistance of OPEN AI's Chat GPT 

# Load necessary libraries
library(reticulate)
library(keras)
library(readxl)
library(dplyr)
library(ggplot2)

# Combat issues with my computer accessing tensorflow
reticulate::py_install("tensorflow", envname = "your_environment_name")
use_virtualenv("myenv", required = TRUE)
py_install("tensorflow", envname = "myenv")
virtualenv_create("myenv")
tensorflow = reticulate::import("tensorflow")
library(tensorflow)

set.seed(1234567890)

# Set Data Path
data_path = "YOUR_PATH/TXEDU17_23.xlsx"
data = read_excel(data_path)

# Set the Dataframe as a Global Environment
attach(data)

# Check the Output to verify data 
nrow(data) 
head(data) 

# Selecting the columns for the model
data_model = data %>%
  select(Student_to_faculty_ratio, Full_Time_Staff_per_Student, Average_salary_of_full_time_professors,
         Percent_of_undergraduate_enrollment_Age_18_to_24,
         Percent_of_full_time_first_time_undergraduates_awarded_any_financial_aid,
         Total_price_for_in_state_students_living_on_campus,
         Percent_of_undergraduate_students_enrolled_exclusively_in_distance_education_courses,
         Percent_admitted, Published_in_state_tuition_and_fees, Books_and_supplies,
         Undergraduate_application_fee, SAT_Math_50th_percentile_score,
         SAT_Reading_and_Writing_50th_percentile_score, Full_time_retention_rate)

# Splitting the data into training and testing sets
indices = sample(1:nrow(data_model), size = 0.7 * nrow(data_model)) 
train_data = data_model[indices, ]
test_data = data_model[-indices, ]

# Normalize the data
train_data_normalized = as.data.frame(scale(train_data))
test_data_normalized = as.data.frame(scale(test_data))

# Prepare the inputs and outputs
train_x = as.matrix(train_data_normalized[, -ncol(train_data_normalized)])
train_y = as.matrix(train_data_normalized[, ncol(train_data_normalized)])
test_x = as.matrix(test_data_normalized[, -ncol(test_data_normalized)])
test_y = as.matrix(test_data_normalized[, ncol(test_data_normalized)])

# Define the neural network model
model = keras_model_sequential() %>%
  layer_dense(units = 64, activation = 'relu', input_shape = c(ncol(train_x))) %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dense(units = 1) 

# Compile the model
model %>% compile(
  loss = 'mse', 
  optimizer = 'rmsprop', 
  metrics = c('mae') 
)

# Fit the model
history = model %>% fit(
  train_x, train_y,
  epochs = 100,
  batch_size = 5,
  validation_split = 0.2
)

# Evaluate the model
model %>% evaluate(test_x, test_y)

# Evaluate the model and directly capture the results
evaluation_results = model %>% evaluate(test_x, test_y)

# Pull the first element of MSE and the second of MAE
nn_mse = evaluation_results[1]
nn_mae = evaluation_results[2]

# Calculate RMSE from MSE
nn_rmse = sqrt(nn_mse)

# Predicting with the neural network model
nn_predictions = model %>% predict(test_x)

# Pull the actual vaules 
actual = test_data$Full_time_retention_rate

# Reversing normalization on data
mean_full_time_retention_rate = mean(train_data$Full_time_retention_rate, na.rm = TRUE)
sd_full_time_retention_rate = sd(train_data$Full_time_retention_rate, na.rm = TRUE)
nn_predictions_unnormalized = (nn_predictions * sd_full_time_retention_rate) + mean_full_time_retention_rate

# Print first few predictions to verify
head(nn_predictions_unnormalized)
head(actual)

# Print MAE, MSE, and RMSE values
print(paste("Neural Network MAE:", nn_mae))
print(paste("Neural Network MSE:", nn_mse))
print(paste("Neural Network RMSE:", nn_rmse))

# Calculate Residuals from actual and predicted values. The calculate Sum of Squared Residuals and Total Sum of Squares.
residuals = actual - nn_predictions_unnormalized
rss_calc = sum(residuals^2)
tss_calc = sum((actual - mean(actual))^2)

# Calculate R Squared and print
r_squared = 1 - (rss_calc / tss_calc)
print(paste("Neural Network R-squared:", r_squared))

# Create a dataframe for plotting
plot_data <- data.frame(
  Actual = actual,
  Predicted = nn_predictions_unnormalized)

# Plot the predicted vs actual retention rates on the testing set
ggplot(plot_data, aes(x = Actual, y = Predicted)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  xlab("Actual Retention Rate") +
  ylab("Predicted  Retention Rate") +
  ggtitle("Neural Network Predictions vs Actual Data (Test Set)") 
