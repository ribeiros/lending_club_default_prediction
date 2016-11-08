
lending_club_2012_2013 <- read.csv("lending_club_rfe/LoanStats3b_securev1.csv", header=TRUE)
lending_club_2007_2011 <- read.csv("lending_club_rfe/LoanStats3a_securev1.csv", header=TRUE)

lending_club_jan_2013 <- subset(lending_club_2012_2013,issue_d == "Jan-2013")
lending_club_feb_2013 <- subset(lending_club_2012_2013,issue_d == "Feb-2013")
lending_club_2012 <- subset(lending_club_2012_2013,grepl("2012",issue_d))
lending_club_final <- rbind(lending_club_jan_2013,lending_club_feb_2013,lending_club_2012,lending_club_2007_2011)
lending_club_final <- subset(lending_club_final,grepl("36",term))

status <- c("Current","Fully Paid","Late (16-30 days)","Does not meet the credit policy. Status:Charged Off","Charged Off","Default","In Grace Period","Late (31-120 days)","Does not meet the credit policy. Status:Fully Paid")
default_status <- c(0,0,1,1,1,1,1,1,0)
lending_club_default <- data.frame(default = default_status[match(lending_club_final$loan_status, status)])
lending_club_final <- cbind(lending_club_final,lending_club_default)

na_pct <-sapply(lending_club_final, function(y) sum(is.na(y))/length(y))
na_pct <- data.frame(na_pct)
mostly_na <- na_pct > 0.2

lending_club_final <- lending_club_final[,mostly_na==FALSE] 

# convert issue_d to dates because as factors they have too many levels!
lending_club_final$issue_d <- as.vector(sapply(lending_club_final$issue_d, function(x) paste0(x,"-01")))
lending_club_final$issue_d <- as.Date(lending_club_final$issue_d,"%b-%Y-%d")

# convert int_rate to numeric
lending_club_final$int_rate <- as.character(lending_club_final$int_rate)
lending_club_final$int_rate <- as.numeric(substr(lending_club_final$int_rate,1,nchar(lending_club_final$int_rate)-1))
 
# convert earliest_cr_line to date
lending_club_final$earliest_cr_line <- as.character(lending_club_final$earliest_cr_line)
lending_club_final$earliest_cr_line <- as.vector(sapply(lending_club_final$earliest_cr_line, function(x) paste0(x,"-01")))
lending_club_final$earliest_cr_line <- as.Date(lending_club_final$earliest_cr_line,"%b-%Y-%d")

# convert revol_util to numeric
lending_club_final$revol_util <- as.character(lending_club_final$revol_util)
lending_club_final$revol_util <- as.numeric(substr(lending_club_final$revol_util,1,nchar(lending_club_final$revol_util)-1))

# convert last_credit_pull_d to dates because as factors they have too many levels!
lending_club_final$last_credit_pull_d <- as.character(lending_club_final$last_credit_pull_d)
lending_club_final$last_credit_pull_d <- as.vector(sapply(lending_club_final$last_credit_pull_d, function(x) paste0(x,"-01")))
lending_club_final$last_credit_pull_d <- as.Date(lending_club_final$last_credit_pull_d,"%b-%Y-%d")
    
# convert zip_code to char because as a factor it has too many levels
#lending_club_final$zip_code <- as.character(lending_club_final$zip_code)
    
# remove useless variables
to_remove <- c("url","desc","title","emp_title","id","loan_status","zip_code")
lending_club_final <- lending_club_final[ , !(names(lending_club_final) %in% to_remove)]


nrows <- nrow(lending_club_final)
ncomplete <- sum(complete.cases(lending_club_final))
print(1-(ncomplete/nrows))

lending_club_final <- lending_club_final[complete.cases(lending_club_final),]

# how many records in data set so far
lcf_before_na_rm <- nrow(lending_club_final)
lcf_before_na_rm

sapply(lending_club_final, function(x) sum(is.na(x)))

lending_club_final <- lending_club_final[, !(colnames(lending_club_final) %in% c("total_rec_int","total_pymnt_inv","total_pymnt"
                                                                                 ,"total_rec_prncp","collection_recovery_fee",
                                                                                 "recoveries","last_pymnt_amnt",
                                                                                 "total_rec_late_fee","last_pymnt_d",
                                                                                 "last_pymnt_amnt","next_pymnt_d"))]

str(lending_club_final)

train_rows <- sample(nrow(lending_club_final),20000)
lending_club.train <- lending_club_final[train_rows,]
lending_club.test <- lending_club_final[-train_rows,]

library(randomForest)
set.seed(9)

rf.fit <- randomForest(as.factor(default)~., data=lending_club.train, importance=TRUE, ntree=700)

varImpPlot(rf.fit)

library(caret)
install.packages("e1071", repos='http://cran.us.r-project.org',dependencies = TRUE)
library(e1071)

# Generate predictions based on model
lending_club.test$default.pred <- predict(rf.fit,lending_club.test)

# Create Confusion Matrix
confusionMatrix(lending_club.test$default.pred,lending_club.test$default)

install.packages("pROC", repos='http://cran.us.r-project.org',dependencies = TRUE)
library(pROC)
# area under a ROC curve
auc(lending_club.test$default,as.numeric(lending_club.test$default.pred))

# Let's use RFE to see if we can prune some variables/features and hopefully get a better result
set.seed(123)
control <- rfeControl(functions=rfFuncs, method="cv", number=10)
lending_club.train.rfe <- lending_club.train[sample(nrow(lending_club.train),2000),]
rfe.train <- rfe(lending_club.train.rfe[,1:42], lending_club.train.rfe[,43], sizes=1:42, rfeControl=control)

# how big is the optimal variable subset?
print(rfe.train$bestSubset)

plot(rfe.train, type=c("g", "o"), cex = 1.0, col = 1:43)

predictors(rfe.train)

# According to RFE the optimal subset of variables is "last_credit_pull_d", "last_fico_range_high", "last_fico_range_low" 
# Redo the Random Forest model with the optimal subset according to RFE
set.seed(44)
rf.fit.opt <- randomForest(as.factor(default)~last_credit_pull_d+last_fico_range_high+last_fico_range_low, 
                                                              data=lending_club.train, ntree=1000, type='classification')

lending_club.test$default.pred.opt <- predict(rf.fit.opt,lending_club.test)

# Create Confusion Matrix
confusionMatrix(lending_club.test$default.pred.opt,lending_club.test$default)

# find area under a ROC curve
auc(lending_club.test$default,as.numeric(lending_club.test$default.pred.opt))

# redo with larger nodesize - should make tree simpler
rf.fit.opt2 <- randomForest(as.factor(default)~last_credit_pull_d+last_fico_range_high+last_fico_range_low, 
                           data=lending_club.train, ntree=1000, type='classification', nodesize=800)

lending_club.test$default.pred.opt2 <- predict(rf.fit.opt2,lending_club.test)

# Create Confusion Matrix
confusionMatrix(lending_club.test$default.pred.opt2,lending_club.test$default)

# find Area Under a ROC Curve (AUC)
auc(lending_club.test$default,as.numeric(lending_club.test$default.pred.opt2))

# BAGGED DECISION TREES
library(rpart)
install.packages("adabag", repos='http://cran.us.r-project.org',dependencies = TRUE)
library(adabag)
lending_club.train$default.factor <- as.factor(lending_club.train$default)
lending_club.test$default.factor <- as.factor(lending_club.test$default)

# mfinal indicates total number of trees grown 
# and minsplit is the minimum number of observations that must exist in a node in order for a split to be attempted
bdt.bagging <- bagging(default.factor~last_credit_pull_d+last_fico_range_high+last_fico_range_low, 
                            data=lending_club.train, mfinal=1000, control=rpart.control(minsplit = 800))

# make predictions
bdt.bagging.pred <- predict.bagging(bdt.bagging, newdata=lending_club.test)

# Create Confusion Matrix
confusionMatrix(bdt.bagging.pred$class,lending_club.test$default)

# find Area Under a ROC Curve (AUC)
auc(lending_club.test$default,as.numeric(bdt.bagging.pred$class))

library(nnet)

lending_club.train.nn <- lending_club.train
lending_club.train.nn$last_credit_pull_d.nn <- as.numeric(as.factor(lending_club.train.nn$last_credit_pull_d))
lending_club.test.nn <- lending_club.test
lending_club.test.nn$last_credit_pull_d.nn <- as.numeric(as.factor(lending_club.test.nn$last_credit_pull_d))

lc.nn.fit <- nnet(default~last_credit_pull_d.nn+last_fico_range_high+last_fico_range_low,
                        data=lending_club.train.nn, size = 2, linout=FALSE)



