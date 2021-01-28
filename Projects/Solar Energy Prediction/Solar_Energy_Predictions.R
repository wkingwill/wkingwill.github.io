library(DataExplorer)
library(skimr)
library(data.table)
library(data.table)
library(dplyr)
library(ggplot2)
library(e1071)
library(randomForest)
library(doParalle)
library(Hmisc)  
library(outlier)
library(knitr)
library(kableExtra)
library(leaflet)
library(lubridate)


#Create folder path
folder_path <- "C:/Users/Eric/OneDrive/IE/R/Project"

#load data sets
solar_dataset <- readRDS(file.path(folder_path, "solar_dataset.RData"));
additional_variables <- readRDS(file.path(folder_path, "additional_variables.RData"));
station_info <- fread(file.path(folder_path, "station_info.csv"))
## create dataframes with input variables

solar_dataset_pred <- as.data.frame(solar_dataset) 
solar_dataset_add_pred <- cbind(as.data.frame(solar_dataset),additional_pred) ##with additional varibales included
dim(solar_dataset_add_pred)
dim(solar_dataset_pred)

###Extract features from Dates, dayofyear, monthofyear

solar_dataset_pred$Date <- as.Date(solar_dataset_pred$Date, format = "%Y%m%d")
solar_dataset_pred<- mutate(solar_dataset_pred, DayofYear = yday(solar_dataset_pred$Date))
solar_dataset_pred <- mutate(solar_dataset_pred, MonthofYear = month(solar_dataset_pred$Date))

head(solar_dataset_pred)

#with additional variables
solar_dataset_add_pred$Date <- as.Date(solar_dataset_add_pred$Date, format = "%Y%m%d")
solar_dataset_add_pred <- mutate(solar_dataset_add_pred, DayofYear = yday(solar_dataset_add_pred$Date))
solar_dataset_add_pred <- mutate(solar_dataset_add_pred, MonthofYear = month(solar_dataset_add_pred$Date))


## split data set
train1 <- head(solar_dataset_pred,5113) ## without additional variables
train2 <- head()
pred_test <- solar_dataset_pred[5114:6909,]
pred_test <- as.data.frame(pred_test)
train1_add <- cbind(train1,additional_pred[,-1])
results <- pred_test[ ,1:99]
head(results)
#### Create train/test/val 

# setting seed to reproduce results of random sampling
set.seed(90) 

# row indices for training data (70%)
train_index <- sample(1:nrow(train1), 0.7*nrow(train1))  

# row indices for validation data (15%)
val_index <- sample(setdiff(1:nrow(train1), train_index), 0.15*nrow(train1))  

# row indices for test data (15%)
test_index <- setdiff(1:nrow(train1), c(train_index, val_index))

# split data
train <- train1[train_index,]
val <- train1[val_index,]
test  <- train1[test_index,]

dim(train1)
dim(train)
dim(val)
dim(test)

sapply(train1, function(x){sum(is.na(x))});



class(train1)





### BASIC PREDICITONS USING SVM #####

pred_columns <- c(names(train1[2:99]))
variable_columns <- c(names(train1[,100:458]))


### TEst on one column ACME

model1 <- svm( x = train1[,variable_columns], y = train1$ACME)


# Get model predictions
predictions_train <- predict(model1, newdata = train[, variable_columns]); 

predictions_test <- predict(model1, newdata = test[,variable_columns]);

# Get errors
errors_train <- predictions_train - train$ACME;
errors_test <- predictions_test - test$ACME;

# Compute Metrics
mse_train <- round(mean(errors_train^2), 2);
mae_train <- round(mean(abs(errors_train)), 2);

mse_test <- round(mean(errors_test^2), 2);
mae_test <- round(mean(abs(errors_test)), 2);

# Build comparison table
comp <- data.table(model = c("standard svm"), 
                         mse_train = mse_train, mae_train = mae_train,
                         mse_test = mse_test, mae_test = mae_test);
comp; 


### predict for results with basic


for (pred_name in pred_columns){
  
  model <- svm( x = train1[,variable_columns], y = train1[,pred_name])
  
  
  
  # Get model predictions
  predictions_result <- predict(model, newdata = pred_test[, variable_columns]); 
  # put into results table 
  results[ ,pred_name] <- predictions_result
  
  
}





#### optimise hyperparameters ####
### GRID search ###

### Define grid
c_values <- 10^seq(from = -2, to = 1, by = 0.5);
eps_values <- 10^seq(from = -2, to = 0, by = 0.5);
gamma_values <- 10^seq(from = -3, to = -1, by = 0.5);

### Compute grid search
grid_results <- data.table();

for (c in c_values){
  for (eps in eps_values){
    for (gamma in gamma_values){
      
      print(sprintf("Start of c = %s - eps = %s - gamma = %s", c, eps, gamma));
      
      # train SVM model with a particular set of hyperparamets
      model2 <- svm( x = train1[,variable_columns], y = train1$ACME,
      cost = c, epsilon = eps, gamma = gamma)
      
      # Get model predictions
      predictions_train <- predict(model2, newdata = train[, variable_columns])
      predictions_val <- predict(model2, newdata = val[, variable_columns])
      
      # Get errors
      errors_train <- predictions_train - train$ACME;
      errors_val <- predictions_val - val$ACME;
      # Compute Metrics
      mse_train <- round(mean(errors_train^2), 2);
      mae_train <- round(mean(abs(errors_train)), 2);
      
      mse_val <- round(mean(errors_val^2), 2);
      mae_val <- round(mean(abs(errors_val)), 2);
      
      # Build comparison table
      grid_results <- rbind(grid_results,
                            data.table(c = c, eps = eps, gamma = gamma, 
                                       mse_train = mse_train, mae_train = mae_train,
                                       mse_val = mse_val, mae_val = mae_val));
    }
  }
}

# View results
View(grid_results);

# Order results by increasing mse and mae
grid_results <- grid_results[order(mse_val, mae_val)];

# Check results
View(grid_results);
grid_results[1]; # Best hyperparameters


# Get optimized hyperparameters
best <- grid_results[1];
best;


# train SVM model with best found set of hyperparamets
model1 <- svm( x = train1[,variable_columns], y = train1$ACME,
             cost = best$c, epsilon = best$eps, gamma = best$gamma)


# Get model predictions
predictions_train <- predict(model1, newdata = train[, variable_columns])
predictions_val <- predict(model1, newdata = val[, variable_columns])
predictions_test <- predict(model1, newdata = test[, variable_columns])

# Get errors
errors_train <- predictions_train - train$ACME
errors_val <- predictions_val - val$ACME
errors_test <- predictions_test - test$ACME

# Compute Metrics
mse_train <- round(mean(errors_train^2), 2);
mae_train <- round(mean(abs(errors_train)), 2);

mse_val <- round(mean(errors_val^2), 2);
mae_val <- round(mean(abs(errors_val)), 2);

mse_test <- round(mean(errors_test^2), 2);
mae_test <- round(mean(abs(errors_test)), 2);

# Build comparison table
comp <- data.table(model = c("optimized_svm"), 
                         mse_train = mse_train, mae_train = mae_train,
                         mse_test = mse_test, mae_test = mae_test);
comp 





#predict for results with optimised parameters 
#(would love to have optimised parameters for each mesa station but my laptop couldnt handle that much computing haha)

for (pred_name in pred_columns){
  
model <- svm( x = train1[,variable_columns], y = train1[,pred_name],
                cost = best$c, epsilon = best$eps, gamma = best$gamma)
  
  
  
# Get model predictions
predictions_result <- predict(model, newdata = pred_test[, variable_columns]); 
# put into results table 
results[ ,pred_name] <- predictions_result

}




write.csv(results,file = file.path(folder_path, "Results.csv"), row.names = FALSE)
