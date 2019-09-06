## Load the required libraries and install missing packages automatically
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(tidyr)) install.packages("tidyr", repos = "http://cran.us.r-project.org")
if(!require(ggthemes)) install.packages("ggthemes", repos = "http://cran.us.r-project.org")
if(!require(readr)) install.packages("readr", repos = "http://cran.us.r-project.org")
if(!require(PRROC)) install.packages("PRROC", repos = "http://cran.us.r-project.org")
if(!require(xgboost)) install.packages("xgboost", repos = "http://cran.us.r-project.org")
if(!require(Matrix)) install.packages("Matrix", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")
if(!require(class)) install.packages("class", repos = "http://cran.us.r-project.org")
if(!require(imbalance)) install.packages("imbalance", repos = "http://cran.us.r-project.org")
if(!require(smotefamily)) install.packages("smotefamily", repos = "http://cran.us.r-project.org")
if(!require(ROSE)) install.packages("ROSE", repos = "http://cran.us.r-project.org")

## Print session information
sessionInfo()

##############################

## STEP 1: IMPORTING DATA

# Credit Card Fraud Detection dataset (Kaggle)
# https://www.kaggle.com/mlg-ulb/creditcardfraud/

# Import the dataset
data <- read_csv("./input/creditcard.csv")
head(data)

##############################

## STEP 2: DATA VISUALIZATION

## Part 2.1: Inspect the dataset
summary(data)
glimpse(data)
names(data)

## Check dimension (number of rows and columns)
dim(data)

## Insights from Part 2.1
# No missing data from summary table
# All variables are masked, except 'Amount', 'Time' and 'Class'

###############

## Part 2.2: Inspect target variable - 'Class'
table(data$Class)

## Plot 'Class' variable
data %>%
  group_by(Class) %>%
  summarise(n = n()) %>%
  mutate(Freq = n / sum(n)) %>%
  ggplot(aes(x= Class, y = Freq, fill= Class)) + 
  geom_bar(stat = "identity", fill="#69b3a2", color="#e9ecef") + 
  geom_text(aes(x = Class, y = Freq, label = round(Freq, digits = 4))) + 
  ggtitle("Distribution of Target ('Class') Variable") + 
  scale_x_continuous(breaks = c(0,1), limits=c(-1,2))

## Convert target variable ('Class') into factor
data$Class <- factor(ifelse(data$Class == 0, "zero", "one")) 

## Insights from Part 2.2
# Highly imbalanced dataset

###############

## Part 2.3: Inspect 'Time' variable

# Convert seconds to hours of the day
data$hour <- (data$Time/3600) %% 24

time_nofraud <- data %>%
  filter(Class == "zero") %>%
  ggplot(aes(x = hour)) + 
  geom_density(fill="#69b3a2", color="#e9ecef") +
  ggtitle("Distribution of 'Time' Variable - Non-Fraud") + 
  scale_x_continuous(breaks = c(0, 4, 8, 12, 16, 20, 24)) +
  theme_ipsum() 

time_fraud <- data %>%
  filter(Class == "one") %>%
  ggplot(aes(x = hour)) + 
  geom_density(fill="#69b3a2", color="#e9ecef") +
  ggtitle("Distribution of 'Time' Variable - Fraud") +
  scale_x_continuous(breaks = c(0, 4, 8, 12, 16, 20, 24)) +
  theme_ipsum() 

grid.arrange(time_nofraud, time_fraud)

###############

## Part 2.4: Inspect 'Amount' variable
 
amt<- data %>% ggplot(aes(x = Amount)) + 
  geom_histogram(fill="#69b3a2", color="#e9ecef") + 
  theme_ipsum() +
  ggtitle("Distribution of 'Amount' Variable")

amt_log <- data %>% ggplot(aes(x = log(Amount))) + 
  geom_histogram(fill="#69b3a2", color="#e9ecef") + 
  theme_ipsum()+
  ggtitle("Distribution of 'Amount' Variable (log)")

grid.arrange(amt, amt_log)

amt_nofraud <- data %>%
  filter(Class == "zero") %>%
  ggplot(aes(x = log(Amount))) + 
  geom_density(fill="#69b3a2", color="#e9ecef") +
  ggtitle("Distribution of 'Amount' Variable - Non-Fraud") + 
  theme_ipsum() + 
  scale_x_continuous(breaks = c(-5, -2.5, 0, 2.5, 5, 7.5, 10), limits=c(-5, 10))

amt_fraud <- data %>%
  filter(Class == "one") %>%
  ggplot(aes(x = log(Amount))) + 
  geom_density(fill="#69b3a2", color="#e9ecef") +
  ggtitle("Distribution of 'Amount' Variable - Fraud") + 
  theme_ipsum()  + 
  scale_x_continuous(breaks = c(-5, -2.5, 0, 2.5, 5, 7.5, 10), limits=c(-5, 10))

grid.arrange(amt_nofraud, amt_fraud)

##############################

## STEP 3: DATA PREPROCESSING

## Part 3.1: Rescale variables to be in the range 0 to 1
predictors <- select(data, -Class)
cbind(melt(apply(predictors, 2, min), value.name = "min"), 
      melt(apply(predictors, 2, max), value.name = "max"))

rescale <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

predictors_rescaled <- as.data.frame(apply(predictors, 2, rescale))
cbind(melt(apply(predictors_rescaled, 2, min), value.name = "min_after_rescaling"), 
      melt(apply(predictors_rescaled, 2, max), value.name = "max_after_rescaling"))

data <- cbind(Class = data$Class, predictors_rescaled)

##############################

## STEP 4: SPLIT DATASET INTO TRAIN & TEST SET

set.seed(1)

###############

## Part 4.1: Baseline dataset
# Train data: 70%; Test data: 30%

train_index <- createDataPartition(data$Class, p=0.7, list=FALSE)

baseline_train <- data[train_index, ]
test <- data[-train_index, ]

train$Class <- factor(train$Class)

###############

## Part 4.2: MWMOTE adjusted dataset
# MWMOTE: Majority Weighted Minority Oversampling Technique
# Oversampling helps to balance class distribution by duplicating minority class instances  

# Generate more noisy instances out of dataset
mwmote_train_gen <- mwmote(as.data.frame(train), numInstances=100000)
table(mwmote_train_gen$Class)

# Bind the train dataset and the newly generated instances
mwmote_train <- rbind(train, newtrain)

# Compare 'Class' variable in the baseline dataset and MWMOTE adjusted dataset
table(baseline_train$Class)
table(mwmote_train$Class)

mwmote_train$Class <- factor(mwmote_train$Class)

## Insights from Part 4.2
# MWMOTE adjusted dataset is a more balanced dataset

###############

## Part 4.3: SMOTE adjusted dataset

# Undersample instances of the majority class so classifiers are not biased toward one class
set.seed(1)
smote_train_gen1 <- SMOTE(X = baseline_train[, -1], target = baseline_train$Class, dup_size = 4)
smote_train_gen2 <- smote_train_gen1$data %>% rename(Class = class)

# Undersample dataset until majority class size matches
smote_train_gen3 <- ovun.sample(Class ~ .,
                                data = smote_train_gen2,
                                method = "under",
                                N = 2 * sum(smote_train_gen2$Class == "one"))

smote_train <- smote_train_gen3$data

# Compare 'Class' variable in the baseline dataset and SMOTE adjusted dataset
table(baseline_train$Class)
table(smote_train$Class)

smote_train$Class <- factor(smote_train$Class)

## Insights from Part 4.3
# SMOTE adjusted dataset is a more balanced dataset

##############################

# STEP 5: MODEL SELECTION & TRAINING

## Evaluation Metric: Area Under the Precision-Recall Curve (AUPRC)

## XGBoost: xgbTree Method

###############

## Part 5.1: XGBoost model for baseline dataset

xgb_baseline_train <- sparse.model.matrix(Class ~ . -1, data = baseline_train)
xgb_mwmote_train <- sparse.model.matrix(Class ~ . -1, data = mwmote_train)
xgb_smote_train <- sparse.model.matrix(Class ~ . -1, data = smote_train)
xgb_test <- sparse.model.matrix(Class ~ . -1, data = test)

ctrl_xgb <- trainControl(method = "cv",
                         number = 3,
                         summaryFunction=prSummary,
                         classProbs=TRUE,
                         allowParallel = TRUE)

set.seed(1)
xgb_baseline_train_benchmark <- train(x = xgb_baseline_train,
                             y = baseline_train$Class,
                             method = "xgbTree",
                             metric = "AUC",
                             trControl = ctrl_xgb)

xgb_baseline_train_pred <- predict(xgb_baseline_train_benchmark, xgb_test, type = "prob")

xgb_baseline_train_scores <- data.frame(event_prob = xgb_baseline_train_pred$one, labels = test$Class)

xgb_baseline_train_auprc <- pr.curve(scores.class0 = xgb_baseline_train_scores[xgb_baseline_train_scores$labels == "one", ]$event_prob,
                                     scores.class1 = xgb_baseline_train_scores[xgb_baseline_train_scores$labels == "zero", ]$event_prob,
                                     curve=T)

xgb_baseline_train_benchmark$bestTune 

paste("Area Under the Precision-Recall Curve:", round(xgb_baseline_train_auprc$auc.integral, 3))

## Add baseline model results to a table for later comparison
XGB_results <- data_frame(Model = "Baseline XGBoost", AUPRC = round(xgb_baseline_train_auprc$auc.integral, 3))
XGB_results %>% knitr::kable()

###############

## Part 5.2: XGBoost model for MWMOTE adjusted dataset

set.seed(1)
xgb_mwmote_train_benchmark <- train(x = xgb_mwmote_train,
                                    y = mwmote_train$Class,
                                    method = "xgbTree",
                                    metric = "AUC",
                                    trControl = ctrl_xgb)


xgb_mwmote_train_pred <- predict(xgb_mwmote_train_benchmark, xgb_test, type = "prob")

xgb_mwmote_train_scores <- data.frame(event_prob = xgb_mwmote_train_pred$one, labels = test$Class)

xgb_mwmote_train_auprc <- pr.curve(scores.class0 = xgb_mwmote_train_scores[xgb_mwmote_train_scores$labels == "one", ]$event_prob, 
                                   scores.class1 = xgb_mwmote_train_scores[xgb_mwmote_train_scores$labels == "zero", ]$event_prob, 
                                   curve=T)

xgb_mwmote_train_benchmark$bestTune 

paste("Area Under the Precision-Recall Curve:", round(xgb_mwmote_train_auprc$auc.integral, 3))

## Add baseline model results to a table for later comparison
XGB_results <- bind_rows(XGB_results,
                         data_frame(Model="MWMOTE Adjusted XGBoost",  
                                    AUPRC = round(xgb_mwmote_train_auprc$auc.integral, 3)))
XGB_results %>% knitr::kable()

###############

## Part 5.3: XGBoost model for SMOTE adjusted dataset

set.seed(1)
xgb_smote_train_benchmark <- train(x = xgb_smote_train,
                                   y = smote_train$Class,
                                   method = "xgbTree",
                                   metric = "AUC",
                                   trControl = ctrl_xgb)


xgb_smote_train_pred <- predict(xgb_smote_train_benchmark, xgb_test, type = "prob")

xgb_smote_train_scores <- data.frame(event_prob = xgb_smote_train_pred$one, labels = test$Class)

xgb_smote_train_auprc <- pr.curve(scores.class0 = xgb_smote_train_scores[xgb_smote_train_scores$labels == "one", ]$event_prob,
                                  scores.class1 = xgb_smote_train_scores[xgb_smote_train_scores$labels == "zero", ]$event_prob,
                                  curve=T)

xgb_smote_train_benchmark$bestTune 

paste("Area Under the Precision-Recall Curve:", round(xgb_smote_train_auprc$auc.integral, 3))

## Add baseline model results to a table for later comparison
XGB_results <- bind_rows(XGB_results,
                         data_frame(Model="SMOTE Adjusted XGBoost",  
                                    AUPRC = round(xgb_smote_train_auprc$auc.integral, 3)))
XGB_results %>% knitr::kable()

##############################