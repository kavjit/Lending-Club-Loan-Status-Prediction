train_main = read.csv('train.csv')
test_main = read.csv('test.csv')

list.of.packages <- c('tibble','plyr','lubridate','gbm')
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
library(tibble)
library(plyr)
library(lubridate)
library(gbm)
#library(modeest)

loan_status = rep('2',times = 253200)

test_main = add_column(test_main, loan_status, .after = 'verification_status')
data2 = rbind(train_main, test_main)
data = data2

x = levels(data$loan_status)
data$loan_status = as.integer(as.character(mapvalues(data$loan_status, from = x, to = c(1,1,0,2))))

##########################################################################################################

#checking for missing values
na_count = sapply(data, function(y) sum(is.na(y)))

###imputing missing values based on column 
#emp length
data$emp_length = as.character(data$emp_length)
data$emp_length[is.na(data$emp_length)] = 'null'
data$emp_length = as.factor(data$emp_length)

#dti
data$dti[is.na(data$dti)] = median(data$dti, na.rm = TRUE)
#revol util
data$revol_util[is.na(data$revol_util)] = median(data$revol_util, na.rm = TRUE)
#mort_acc
data$mort_acc[is.na(data$mort_acc)] = 2
#pub rec bankruptcies
data$pub_rec_bankruptcies[is.na(data$pub_rec_bankruptcies)] = 0

##########################################################################################################

###dropping some columns
drop = c('emp_title','zip_code')
data = data[,!(names(data)%in%drop)]


###formatting date column as age
data$earliest_cr_line_date = parse_date_time(data$earliest_cr_line, orders = 'my')
reference_date = parse_date_time('01-Jan-2007', orders = 'dmy')
x = interval(reference_date, data$earliest_cr_line_date)
data$credit_age = abs(x %/% months(1))
drop = c('earliest_cr_line','earliest_cr_line_date')
data = data[,!(names(data)%in%drop)]


# ###Converting title column text properly
# data$title = tolower(data$title)
# x = data$title
# x[grep('consolidation',x, value = FALSE)] = 'consolidation'
# x[grep('credit',x, value = FALSE)] = 'consolidation'
# x[grep('cc',x, value = FALSE)] = 'consolidation'
# x[grep('consolidate',x, value = FALSE)] = 'consolidation'
# x[grep('consol',x, value = FALSE)] = 'consolidation'
# x[grep('consolodation',x, value = FALSE)] = 'consolidation'
# x[grep('debt',x, value = FALSE)] = 'consolidation'
# x[grep('loan',x, value = FALSE)] = 'consolidation'
# x[grep('refinance',x, value = FALSE)] = 'consolidation'
# x[grep('refi',x, value = FALSE)] = 'consolidation'
# x[grep('payoff',x, value = FALSE)] = 'bills'
# x[grep('pay off',x, value = FALSE)] = 'bills'
# x[grep('bills',x, value = FALSE)] = 'bills'
# x[grep('pay',x, value = FALSE)] = 'bills'
# x[grep('home',x, value = FALSE)] = 'home'
# x[grep('house',x, value = FALSE)] = 'home'
# x[grep('kitchen',x, value = FALSE)] = 'home'
# x[grep('roof',x, value = FALSE)] = 'home'
# x[grep('tub',x, value = FALSE)] = 'home'
# x[grep('moving',x, value = FALSE)] = 'home'
# x[grep('relocation',x, value = FALSE)] = 'home'
# x[grep('medical',x, value = FALSE)] = 'medical'
# x[grep('business',x, value = FALSE)] = 'business'
# x[grep('car',x, value = FALSE)] = 'car'
# x[grep('auto',x, value = FALSE)] = 'car'
# x[grep('motorcycle',x, value = FALSE)] = 'car'
# x[grep('truck',x, value = FALSE)] = 'car'
# x[grep('wedding',x, value = FALSE)] = 'wedding'
# x[grep('ring',x, value = FALSE)] = 'wedding'
# x[grep('personal',x, value = FALSE)] = 'personal'
# x[grep('purchase',x, value = FALSE)] = 'purchase'
# x[grep('holiday',x, value = FALSE)] = 'vacation'
# x[grep('vacation',x, value = FALSE)] = 'vacation'
# 
# categories = c('consolidation','bills','home','medical','business','car','wedding','personal','purchase','vacation')
# x[!x%in%categories] = 'other'
# y = data.frame(table(x))
# data$title = as.factor(x)

#trying to drop or keep title
drop = c('title','grade')
data = data[,!(names(data)%in%drop)]

na_count = sapply(data, function(y) sum(is.na(y)))


###Converting ordinal variables to numeric
# x = levels(data$sub_grade)
# data$sub_grade = as.integer(as.character(mapvalues(data$sub_grade, from = x, to = seq(35,1))))


########## EDA ##########################################################################
numerical = unlist(lapply(data, is.numeric)) #for numerical variables
data_num = data[, numerical]
correlations  = as.data.frame(round(cor(data_num),2))

#########################################################################################



###Preparation for one hot encoding
#datatypes of all columns
types = data.frame(sapply(data, class))
###One hot encoding
#make the loan status the last column - y
data$y = data$loan_status
drop = c('loan_status')
data = data[,!(names(data)%in%drop)]
#one hot encoding
fake.y = rep(0, length(data[,1]))
tmp = model.matrix(~.,data = data)
tmp = data.frame(tmp[, -1])  # remove the 1st column (the intercept) of tmp
#write.csv(tmp,'tmp.csv')
data3 = tmp


###Splitting data again to test and train
train = data3[1:nrow(train_main),]
test = data3[(nrow(train_main)+1):nrow(data3),] #note - train and test both have the id column, doesn't matter
                                                #since df after pca is used for model

#################### Winsorization ####################################
###winsorizing highly correlated variables
winsorize = function(vec,pr1,pr2){
  x = quantile(vec,probs=c(pr1,pr2))
  vec[vec>x[2]] = x[2]
  vec[vec<x[1]] = x[1]
  return(vec)
}

#making one dti outlier 0
train$dti[train$dti==-1] = 0

# #train$int_rate = winsorize(train$int_rate,0.01,0.99)
# train$dti = winsorize(train$dti,0.01,0.99)
# train$revol_util = winsorize(train$revol_util,0.01,0.99)
# train$loan_amnt = winsorize(train$loan_amnt,0.01,0.99)
# train$annual_inc = winsorize(train$annual_inc,0.01,0.99)
# train$credit_age = winsorize(train$credit_age,0.01,0.99)
# train$total_acc = winsorize(train$total_acc,0.01,0.99)
# train$pub_rec_bankruptcies = winsorize(train$pub_rec_bankruptcies,0.01,0.99)
# train$pub_rec = winsorize(train$pub_rec,0.01,0.99)


############################# log transforms ############################
####for train
train$annual_inc = log(1 + train$annual_inc)
train$revol_bal = log(1 + train$revol_bal)
train$revol_util = log(1 + train$revol_util)
train$pub_rec_bankruptcies = log(1 + train$pub_rec_bankruptcies)
train$mort_acc = log(1 + train$mort_acc)
train$open_acc = log(1 + train$open_acc)
train$fico_range_low = log(1 + train$fico_range_low)
train$fico_range_high = log(1 + train$fico_range_high)

####for test
test$annual_inc = log(1 + test$annual_inc)
test$revol_bal = log(1 + test$revol_bal)
test$revol_util = log(1 + test$revol_util)
test$pub_rec_bankruptcies = log(1 + test$pub_rec_bankruptcies)
test$mort_acc = log(1 + test$mort_acc)
test$open_acc = log(1 + test$open_acc)
test$fico_range_low = log(1 + test$fico_range_low)
test$fico_range_high = log(1 + test$fico_range_high)




########################################################################################
###PCA to data3
#remove dependent var create new df's without Y
# drop = c('y','id')
# train_pca = train[,!(names(train)%in%drop)]
# test_pca = test[,!(names(test)%in%drop)]
# 
# #applying pca
# pca = prcomp(train_pca, center = TRUE, scale. = TRUE)
# summary(pca)
# plot(pca, type='l')
# 
# #creating new train df
# train_pca_comp = data.frame(Y = train$y, pca$x) #joining Y variable back
# train_pca_comp = train_pca_comp[,1:102] #taking first 75 components
# 
# #transforming test accordingly - this avoids data leakage
# test_pca_comp = as.data.frame(predict(pca, newdata = test_pca))
# test_pca_comp = test_pca_comp[,1:101]

#train_pca_comp and test_pca_comp are used for the model fitting and prediction
##########################################################################################################

###IF NOT DOING PCA
drop = c('y','id')
train_comp = train[,!(names(train)%in%drop)]
test_comp = test[,!(names(test)%in%drop)]
train_comp = data.frame(Y = train$y, train_comp)

##########################################################################################################


############################# FITTING MODELS #############################
#LOGISTIC REGRESSION
model1 = glm(Y~., data=train_comp, family=binomial)
ypred.model1 = predict(model1, newdata = test_comp, type = 'response')


#XGBOOST
# model2 <- gbm(
#   formula = Y ~ .,
#   distribution = "gaussian",
#   data = train_comp,
#   n.trees = 483,
#   interaction.depth = 5,
#   shrinkage = 0.1,
#   n.minobsinnode = 5,
#   bag.fraction = .65, 
#   train.fraction = 1,
#   n.cores = NULL, # will use all cores by default
#   verbose = FALSE
# )
# 
# ypred.model2 = predict(model2, test_comp, n.trees = 483)



#OUTPUT 1
output = data.frame(id = test$id,prob = ypred.model1)
write.csv(output,'mysubmission1.txt',row.names = FALSE)

#OUTPUT 2
# output = data.frame(id = test$id,prob = ypred.model2)
# write.csv(output,'mysubmission2.txt',row.names = FALSE)


