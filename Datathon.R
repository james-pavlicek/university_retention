# Welcome to my Datathon Project Submission, by James Pavlicek.
# Project Title: Predicting Texas Public Universities Retention Rate With Multi Variable Linear Regression and Neural Networks in R
# This Project and it's supporting files were built from Feb 15th to Feb 16th for use in Texas State University's Datathon Competition.
# The code has received an additional revision on Feb 19th with comments to provide easier reader interpretation.
# This code is free use and anyone can use it.
# This code has two parts. You are currently reading "Datathon.R" which contains the code for the Multi Variable Linear Regression.
# The other part is "Datathon_2.R" which will utilize Neural Networks for a similar analysis.
# The last document you will need is "TXEDU17_23" which can be found at my github: https://github.com/james-pavlicek
# I hope you enjoy and reach out me if you have any questions at all! (https://www.jamespavlicek.com/) (https://www.linkedin.com/in/jamespavlicek/) 


# James Pavlicek / 2/16/2024 / Datathon / Multi Varaible Linear Regression

# Load necessary libraries
library(readxl)
library(ggplot2)

set.seed(1234567890)

# Set Data Path
data_path = "YOUR_PATH/TXEDU17_23.xlsx"
dat = read_excel(data_path)

# Set the Dataframe as a Global Environment
attach(dat)

# Check the Output to verify data 
nrow(dat) 
head(dat) 

# Use pairs() to display a scatter plot matrix
selected_data = dat[, c(5:17,18)]
pairs(selected_data)


# Calculate and output correlations for Full_time_retention_rate
correlation_matrix = cor(dat)
correlation_with_last_var = correlation_matrix[, ncol(correlation_matrix)]
correlation_df <- data.frame(Variable = names(correlation_with_last_var), Correlation = correlation_with_last_var)
print(correlation_df)


# Split the data set (70% training / 30% test) 
training_size = ceiling(dim(dat)[1]*0.70)
index = sample(x = 1:dim(dat)[1], size = training_size, replace = FALSE) 
training_set = dat[index,]
testing_set = dat[-index,] 

# Linear regression testing of each independent variable individually to see which are impactful
summary(lm(Full_time_retention_rate ~ Student_to_faculty_ratio , data = dat)) #
summary(lm(Full_time_retention_rate ~ Full_Time_Staff_per_Student , data = dat))
summary(lm(Full_time_retention_rate ~ Average_salary_of_full_time_professors , data = dat)) #
summary(lm(Full_time_retention_rate ~ Percent_of_undergraduate_enrollment_Age_18_to_24 , data = dat))#
summary(lm(Full_time_retention_rate ~ Percent_of_full_time_first_time_undergraduates_awarded_any_financial_aid , data = dat))
summary(lm(Full_time_retention_rate ~ Total_price_for_in_state_students_living_on_campus , data = dat))#
summary(lm(Full_time_retention_rate ~ Percent_of_undergraduate_students_enrolled_exclusively_in_distance_education_courses , data = dat))
summary(lm(Full_time_retention_rate ~ Percent_admitted , data = dat))
summary(lm(Full_time_retention_rate ~ Published_in_state_tuition_and_fees , data = dat)) #
summary(lm(Full_time_retention_rate ~ Books_and_supplies , data = dat))
summary(lm(Full_time_retention_rate ~ Undergraduate_application_fee , data = dat))
summary(lm(Full_time_retention_rate ~ SAT_Math_50th_percentile_score , data = dat)) #
summary(lm(Full_time_retention_rate ~ SAT_Reading_and_Writing_50th_percentile_score , data = dat)) #
summary(lm(Full_time_retention_rate ~ Full_time_retention_rate , data = dat)) #

# Intial model with only the varaibles deemed impactful from the testing above
linear_model_first = lm(Full_time_retention_rate ~ Student_to_faculty_ratio +
                    Average_salary_of_full_time_professors + 
                    Percent_of_undergraduate_enrollment_Age_18_to_24 + 
                    Percent_of_full_time_first_time_undergraduates_awarded_any_financial_aid +
                    Published_in_state_tuition_and_fees + 
                    SAT_Math_50th_percentile_score + 
                    SAT_Reading_and_Writing_50th_percentile_score, data = training_set) 

summary(linear_model_first)


#Second Model will all the independent variables
linear_model = lm(Full_time_retention_rate ~ Student_to_faculty_ratio +
                    Full_Time_Staff_per_Student +
                    Average_salary_of_full_time_professors + 
                    Percent_of_undergraduate_enrollment_Age_18_to_24 + 
                    Percent_of_full_time_first_time_undergraduates_awarded_any_financial_aid +
                    Total_price_for_in_state_students_living_on_campus + 
                    Percent_of_undergraduate_students_enrolled_exclusively_in_distance_education_courses + 
                    Percent_admitted + 
                    Published_in_state_tuition_and_fees + 
                    Books_and_supplies + 
                    Undergraduate_application_fee + 
                    SAT_Math_50th_percentile_score + 
                    SAT_Reading_and_Writing_50th_percentile_score , data = training_set) 

summary(linear_model)


# Predicting the full-time retention rate using the linear model on the testing set
predictions = predict(linear_model, newdata=testing_set)
print(predictions)

# Extract actual retention rates from the testing set
actual = testing_set$Full_time_retention_rate
predicted = predictions

# Calculate Mean Absolute Error, Mean Squared Error, Root Mean Squared Error
mae_calc = mean(abs(actual - predicted), na.rm = TRUE)
mse_calc = mean((actual - predicted)^2, na.rm = TRUE)
rmse_calc = sqrt(mse_calc)

# Print MAE, MSE, and RMSE values
print(paste("Multi Linear Regression MAE:", mae_calc))
print(paste("Multi Linear Regression MSE:", mse_calc))
print(paste("Multi Linear Regression RMSE:", rmse_calc))

# Calculate Residuals from actual and predicted values. The calculate Sum of Squared Residuals and Total Sum of Squares.
residuals = actual - predicted
rss_calc = sum(residuals^2)
tss_calc = sum((actual - mean(actual))^2)

# Calculate R Squared and print
r_squared = 1 - (rss_calc / tss_calc)
print(paste("Multi Linear Regression R-squared:", r_squared))

# Generate predictions on the testing set
testing_set$predicted_retention_rate = predict(linear_model, newdata = testing_set)

# Plot the predicted vs actual retention rates on the testing set
ggplot(testing_set, aes(x = Full_time_retention_rate, y = predicted_retention_rate)) +
  geom_point() +
  geom_smooth(method = lm, se = FALSE) +
  xlab("Actual Retention Rate") +
  ylab(" Predicted Retention Rate") +
  ggtitle("Multi Linear Regression Predictions vs Actual Data (Test Set)")