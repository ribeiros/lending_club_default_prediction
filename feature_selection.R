# create dataset
lending_club_2012_2013 <- read.csv("LoanStats3b_sam.csv", header=TRUE)
lending_club_2007_2011 <- read.csv("LoanStats3a_sam.csv", header=TRUE)
lending_club_jan_2013 <- lending_club_2012_2013[lending_club_2012_2013[,16]=="Jan-2013",]
lending_club_feb_2013 <- lending_club_2012_2013[lending_club_2012_2013[,16]=="Feb-2013",]
lending_club_2012 <- subset(lending_club_2012_2013,grepl("2012",issue_d))
lending_club_final <- rbind(lending_club_jan_2013,lending_club_feb_2013,lending_club_2012,lending_club_2007_2011)
lending_club_final <- subset(lending_club_final,grepl("36",term))

# add label (default_status) to dataset
status <- c("Current","Fully Paid","Late (16-30 days)","Does not meet the credit policy. Status:Charged Off","Charged Off","Default","In Grace Period","Late (31-120 days)","Does not meet the credit policy. Status:Fully Paid")
default_status <- c(0,0,1,1,1,1,1,1,0)
lending_club_default <- data.frame(default = default_status[match(lending_club_final$loan_status, status)])
lending_club_final <- cbind(lending_club_final,lending_club_default)
str(lending_club_final)

int_rate_pct <- as.character(lending_club_final$int_rate)
int_rate_pct <- as.numeric(substr(int_rate_pct,1,nchar(int_rate_pct)-1))

#Remove redundant features - http://machinelearningmastery.com/feature-selection-with-the-caret-r-package/
########################################################################################################
# ensure the results are repeatable
set.seed(7)
# load the library
library(mlbench)
library(caret)

lending_club_final_fact_as_num <- lending_club_final
indx <- sapply(lending_club_final_fact_as_num, is.factor) # http://stackoverflow.com/questions/27528907/how-to-convert-data-frame-column-from-factor-to-numeric
#lending_club_final_fact_as_num[indx] <- lapply(lending_club_final_fact_as_num[indx], function(x) as.numeric(as.character(x)))
lending_club_final_fact_as_num[indx] <- lapply(lending_club_final_fact_as_num[indx], function(x) seq_along(levels(x))[x])
str(lending_club_final_fact_as_num)

num_feat <- sapply(lending_club_final_fact_as_num, is.numeric)
lending_club_final_num_only <- lending_club_final_fact_as_num[ , num_feat] #http://stackoverflow.com/questions/5863097/selecting-only-numeric-columns-from-a-data-frame
str(lending_club_final_num_only)

# at this point I exported lending_club_final_num_only to csv and manually removed features with mostly NULL values 
# as well as records with NULL values and re-imported
lending_club_final_num_only_clean <- read.csv("lending_club_final_num_only_clean.csv")

########################################################################################################################################
# Remove variables/features which are highly correlated - http://machinelearningmastery.com/feature-selection-with-the-caret-r-package/
# calculate correlation matrix
correlationMatrix <- cor(lending_club_final_num_only_clean)
# summarize the correlation matrix
print(correlationMatrix)
# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.5)
# print indexes of highly correlated attributes
print(highlyCorrelated)
library(corrplot)
corrplot(correlationMatrix, method="circle")

#Remove some features due to correlation:
# At this point I manually removed the highly correlated variables from lending_club_final_num_only_clean.csv and
# saved as lending_club_final_num_only_clean_rem_corr.csv
lending_club_final_num_only_clean_rem_corr <- read.csv("lending_club_final_num_only_clean_rem_corr.csv")

# visualize the correlation matrix
correlationMatrix <- cor(lending_club_final_num_only_clean_rem_corr)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.5)
corrplot(correlationMatrix, method="circle")

######################################################################################################################
#Feature Selection - Boruta -- https://www.analyticsvidhya.com/blog/2016/03/select-important-variables-boruta-package/
install.packages("Boruta")
library("Boruta")

# check that all features/variables are numeric
summary(lending_club_final_num_only_clean_rem_corr)
lending_club_final_num_only_clean_rem_corr[is.na(lending_club_final_num_only_clean_rem_corr)]
lending_club_final_num_only_clean_rem_corr[lending_club_final_num_only_clean_rem_corr==""]

lending_club_final.train <- lending_club_final_num_only_clean_rem_corr[sample(nrow(lending_club_final_num_only_clean_rem_corr),10000),]
# remove features/variables which are only known after default
lending_club_final.train <- lending_club_final.train[, !(colnames(lending_club_final.train) %in% c("collection_recovery_fee","recoveries","last_pymnt_amnt","total_rec_late_fee","last_pymnt_d","last_pymnt_amnt","next_pymnt_d"))]

set.seed(129)
boruta.train <- Boruta(default_status~., data = lending_club_final.train, doTrace = 2)
print(boruta.train$finalDecision)
print(boruta.train)

plot(boruta.train, xlab = "", xaxt = "n")
lz<-lapply(1:ncol(boruta.train$ImpHistory),function(i) boruta.train$ImpHistory[is.finite(boruta.train$ImpHistory[,i]),i])
names(lz) <- colnames(boruta.train$ImpHistory)
Labels <- sort(sapply(lz,median))
axis(side = 1,las=2,labels = names(Labels), at = 1:ncol(boruta.train$ImpHistory), cex.axis = 0.7)

final.boruta <- TentativeRoughFix(boruta.train)
print(final.boruta$finalDecision)
getSelectedAttributes(final.boruta, withTentative = F)

## Do the same with RFE
library(caret)
library(randomForest)
set.seed(123)
control <- rfeControl(functions=rfFuncs, method="cv", number=10)
lending_club_final.train <- lending_club_final.train[sample(nrow(lending_club_final.train),2000),]
rfe.train <- rfe(lending_club_final.train[,1:28], lending_club_final.train[,29], sizes=1:28, rfeControl=control)
print(rfe.train)
print(rfe.train$optsize)
print(rfe.train$bestSubset)
print(rfe.train$optVariables)

plot(rfe.train, type=c("g", "o"), cex = 1.0, col = 1:28)
predictors(rfe.train)

# estimate variable importance
importance <- varImp(rfe.train, scale=FALSE)
# summarize importance
print(importance)
# plot importance
plot(importance)
