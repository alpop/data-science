Start.time=Sys.time()
# 
# Kaggle competition:
# "Quantify property hazards before time of inspection" 
# https://www.kaggle.com/c/liberty-mutual-group-property-inspection-prediction
#
# Rank: 121st (out of 2236)
#

# Load required libraries

require(xgboost)
require(caret)
require(plyr)
require(dummies)

# Gini evaluaton metric
# The following piece of code copied from Chippy's script:
# https://www.kaggle.com/nigelcarpenter

SumModelGini = function(solution, submission) {
        df = data.frame(solution = solution, submission = submission)
        df = df[order(df$submission, decreasing = TRUE),]
        df$random = (1:nrow(df))/nrow(df)
        totalPos  = sum(df$solution)
        df$cumPosFound = cumsum(df$solution)       # this will store the cumulative number of positive examples found (used for computing "Model Lorentz")
        df$Lorentz =  df$cumPosFound / totalPos    # this will store the cumulative proportion of positive examples found ("Model Lorentz")
        df$Gini = df$Lorentz - df$random           # will store Lorentz minus random
        return(sum(df$Gini))
}

NormalizedGini = function(solution, submission) {
        SumModelGini(solution, submission) / SumModelGini(solution, solution)
}

# Wrap up into a function to be called within xgboost.train

evalgini  = function(preds, dtrain) {
        labels = getinfo(dtrain, "label")
        err = NormalizedGini(as.numeric(labels),as.numeric(preds))
        return(list(metric = "Gini", value = err))
}

# Build function for xgboost prediction
# This function is based on "Blah - XGB"  scipt:
# https://www.kaggle.com/soutik

xgb_pred  =  function(train=train.data,test=test.data,seed=seed,
                     param=param,nround=nround,print_n=print_n,
                     early_stop=early_stop,feval=evalgini,maximize = T) {

        # Split data into training and cross-validation sets 
        
        set.seed(seed) 
        inTrain=4001:50999
        trn = train[ inTrain,]
        cv =  train[-inTrain,]
         
        label = as.numeric(subset(trn, select=Hazard)$Hazard)
        label.cv = as.numeric(subset(cv, select=Hazard)$Hazard)
        data = as.matrix(subset(trn, select=-c(Id,Hazard)))                   
        data.cv = as.matrix(subset(cv, select=-c(Id,Hazard)))
        data.test = as.matrix(subset(test, select=-Id))

        data1  = data
        data1.cv = data.cv
        label1 = label
        label1.cv = label.cv
        
        # Call xgboost training then predict 1.
        
        xgtrain=xgb.DMatrix(data=data,label=label)      
        xgval=xgb.DMatrix(data=data.cv,label=label.cv) 
        watchlist = list(val=xgval, train=xgtrain)
        cat ('Boosting trees 1 (of 2)...\n')
        
        model_ = xgb.train(param, xgtrain, nround, watchlist,feval = feval, 
                            print.every.n = print_n,early.stop.round = early_stop, 
                            maximize=maximize);
        cat(sprintf ('\nTraining 1 (of 2) finished. Best Score: %5.5f\n',model_$bestScore)) 
        cat('Predicting...\n')
        
        test.pred1  = predict(model_,xgb.DMatrix(data.test),ntreelimit=model_$bestInd)
        train.pred1 = predict(model_,xgb.DMatrix(data),ntreelimit=model_$bestInd)
        cv.pred1    = predict(model_,xgb.DMatrix(data.cv),ntreelimit=model_$bestInd)
       
        cat(sprintf('Normalized Gini of the  model 1 on train data:  %5.5f\n', NormalizedGini(label,train.pred1)))
        cat(sprintf('Normalized Gini of the  model 1 on cv    data:  %5.5f\n', NormalizedGini(label.cv,cv.pred1)))
         
        # Call xgboost training then predict 2.
        
        inTrain=(50999-4000):1
        trn = train[ inTrain,]
        cv =  train[-inTrain,]
        
        lab    = (as.numeric(subset(trn, select=Hazard)$Hazard))
        lab.cv = (as.numeric(subset(cv, select=Hazard)$Hazard))
        label  = log(as.numeric(subset(trn, select=Hazard)$Hazard))
        label.cv = log(as.numeric(subset(cv, select=Hazard)$Hazard))
        data = as.matrix(subset(trn, select=-c(Id,Hazard)))                   
        data.cv = as.matrix(subset(cv, select=-c(Id,Hazard)))
        data.test = as.matrix(subset(test, select=-Id)) 
           
        data2  = data
        data2.cv = data.cv
        label2 = label
        label2.cv = label.cv    
        
        xgtrain=xgb.DMatrix(data=data,label=label)      
        xgval=xgb.DMatrix(data=data.cv,label=label.cv) 
        
        watchlist = list(val=xgval, train=xgtrain)
        cat ('Boosting trees 2 (of 2)...\n')
       
        model_ = xgb.train(param, xgtrain, nround, watchlist,feval = feval, 
                            print.every.n = print_n,early.stop.round = early_stop, 
                            maximize=maximize);
        cat(sprintf ('\nTraining 2 (of 2) finished. Best Score: %5.5f\n',model_$bestScore))
        
        cat('Predicting...\n')
        test.pred2  = predict(model_,xgb.DMatrix(data.test),ntreelimit=model_$bestInd)
        train.pred2 = predict(model_,xgb.DMatrix(data),ntreelimit=model_$bestInd)
        cv.pred2    = predict(model_,xgb.DMatrix(data.cv),ntreelimit=model_$bestInd)
        cat('Predicting finished\n')
        cat(sprintf('Normalized Gini of the  model 2 on train data:  %5.5f\n', NormalizedGini(lab,train.pred2)))
        cat(sprintf('Normalized Gini of the  model 2 on cv    data:  %5.5f\n', NormalizedGini(lab.cv,cv.pred2)))
   
        # Build mixed model 
        
        pred  =  1.4 * test.pred1 + 8.6 * test.pred2
        
        ret.list=list(pred,inTrain)
        names(ret.list)=c('pred','inTrain')
        return(ret.list)
}

# Read competition data

train.data = read.csv("./train.csv", quote="")
test.data = read.csv("./test.csv", quote="")

# MODEL 1.

# Drop unimportant vars
# The idea from "Blah - XGB"  script here:
# https://www.kaggle.com/soutik/

train1.data = subset(train.data, select=-c(T1_V10,T1_V13,T2_V7,T2_V10))
test1.data  = subset(test.data,  select=-c(T1_V10,T1_V13,T2_V7,T2_V10))

train2.data=train1.data
test2.data=test1.data

# Feature engineering: enumerate input values starting from 0

for (i in 3:length(colnames(train1.data))) {
        train1.data[,i]=as.integer(mapvalues(train1.data[,i],levels(as.factor(train1.data[,i])),
                                         1:length(levels(as.factor(train1.data[,i])))))-1
}
for (i in 2:length(colnames(test1.data))) {
        test1.data[,i]=as.integer(mapvalues(test1.data[,i],levels(as.factor(test1.data[,i])),
                                            1:length(levels(as.factor(test1.data[,i])))))-1
}

# Set parameters for the model

seed=9001; 
nround = 20000; print_n = 200; early_stop = 200
param =  list("objective" = "reg:linear",
              "eta" = 0.005,
              "min_child_weight" = 6, 
              "subsample" = .7, 
              "colsample_bytree" = .7, 
              "scale_pos_weight" = 1.0,
              "max_depth" = 9,
              "silent" = 1,"nthread" = 2)

cat('MODEL 1. Building...\n')
print(param)
call1=xgb_pred(train=train1.data,test=test1.data,seed=seed,
               param=param,nround=nround,print_n=print_n,early_stop=early_stop,
               feval=NULL,maximize = F)
pred1       = call1$pred

# MODEL 2.

train2.data= dummy.data.frame(train2.data,fun=as.numeric)
test2.data = dummy.data.frame(test2.data,fun=as.numeric)

cat('MODEL 2. Building...\n')

call2=xgb_pred(train=train2.data,test=test2.data,seed=seed,
              param=param,nround=nround,print_n=print_n,early_stop=early_stop,
              feval=NULL,maximize = F)

pred2=call2$pred

# MODEL 3.

# The following feature engineering is of my own

dtrain = train.data
dtest  =  test.data

dtrain$T1_V1=as.numeric(dtrain$T1_V1)
dtrain$T1_V2=as.numeric(dtrain$T1_V2)
dtrain$T1_V3=as.numeric(dtrain$T1_V3)
dtrain$T1_V10=as.numeric(dtrain$T1_V10)  
dtrain$T1_V13=as.numeric(dtrain$T1_V13)
dtrain$T1_V14=as.numeric(dtrain$T1_V14)
dtrain$T2_V1=as.numeric(dtrain$T2_V1)
dtrain$T2_V2=as.numeric(dtrain$T2_V2)
dtrain$T2_V4=as.numeric(dtrain$T2_V4)
dtrain$T2_V6=as.numeric(dtrain$T2_V6)
dtrain$T2_V7=as.numeric(dtrain$T2_V7)
dtrain$T2_V8=as.numeric(dtrain$T2_V8)
dtrain$T2_V9=as.numeric(dtrain$T2_V9)
dtrain$T2_V10=as.numeric(dtrain$T2_V10)
dtrain$T2_V14=as.numeric(dtrain$T2_V14)
dtrain$T2_V15=as.numeric(dtrain$T2_V15)

dtrain$T1_V4m=match(dtrain$T1_V4,LETTERS)
dtrain$T1_V5m=match(dtrain$T1_V5,LETTERS)
dtrain$T1_V6m=match(dtrain$T1_V6,LETTERS)
dtrain$T1_V7m=match(dtrain$T1_V7,LETTERS)
dtrain$T1_V8m=match(dtrain$T1_V8,LETTERS)
dtrain$T1_V9m=match(dtrain$T1_V9,LETTERS)
dtrain$T1_V11m=match(dtrain$T1_V11,LETTERS)
dtrain$T1_V12m=match(dtrain$T1_V12,LETTERS)
dtrain$T1_V15m=match(dtrain$T1_V15,LETTERS)
dtrain$T1_V16m=match(dtrain$T1_V16,LETTERS)
dtrain$T1_V17m=(match(dtrain$T1_V17,LETTERS)-match("N",LETTERS))/11
dtrain$T2_V5m=match(dtrain$T2_V5,LETTERS)
dtrain$T2_V12m=(match(dtrain$T2_V12,LETTERS)-match("N",LETTERS))/11
dtrain$T2_V13m=match(dtrain$T2_V13,LETTERS)

dtest$T1_V1=as.numeric(dtest$T1_V1)
dtest$T1_V2=as.numeric(dtest$T1_V2)
dtest$T1_V3=as.numeric(dtest$T1_V3)
dtest$T1_V10=as.numeric(dtest$T1_V10)  
dtest$T1_V13=as.numeric(dtest$T1_V13)
dtest$T1_V14=as.numeric(dtest$T1_V14)
dtest$T2_V1=as.numeric(dtest$T2_V1)
dtest$T2_V2=as.numeric(dtest$T2_V2)
dtest$T2_V4=as.numeric(dtest$T2_V4)
dtest$T2_V6=as.numeric(dtest$T2_V6)
dtest$T2_V7=as.numeric(dtest$T2_V7)
dtest$T2_V8=as.numeric(dtest$T2_V8)
dtest$T2_V9=as.numeric(dtest$T2_V9)
dtest$T2_V10=as.numeric(dtest$T2_V10)
dtest$T2_V14=as.numeric(dtest$T2_V14)
dtest$T2_V15=as.numeric(dtest$T2_V15)

dtest$T1_V4m=match(dtest$T1_V4,LETTERS)
dtest$T1_V5m=match(dtest$T1_V5,LETTERS)
dtest$T1_V6m=match(dtest$T1_V6,LETTERS)
dtest$T1_V7m=match(dtest$T1_V7,LETTERS)
dtest$T1_V8m=match(dtest$T1_V8,LETTERS)
dtest$T1_V9m=match(dtest$T1_V9,LETTERS)
dtest$T1_V11m=match(dtest$T1_V11,LETTERS)
dtest$T1_V12m=match(dtest$T1_V12,LETTERS)
dtest$T1_V15m=match(dtest$T1_V15,LETTERS)
dtest$T1_V16m=match(dtest$T1_V16,LETTERS)
dtest$T1_V17m=(match(dtest$T1_V17,LETTERS)-match("N",LETTERS))/11
dtest$T2_V5m=match(dtest$T2_V5,LETTERS)
dtest$T2_V12m=(match(dtest$T2_V12,LETTERS)-match("N",LETTERS))/11
dtest$T2_V13m=match(dtest$T2_V13,LETTERS)


train3 = subset(dtrain, select=c(T1_V1,T1_V2,T1_V3,T1_V10,T1_V13,T1_V14,T2_V1,T2_V2,T2_V4,T2_V6,T2_V7,T2_V9,T2_V10,
                                 T2_V14,T2_V15,T1_V4m,T1_V5m,T1_V7m,T1_V8m,T1_V9m,T1_V11m,T1_V12m,T1_V15m,T1_V16m,
                                 T2_V5m,T2_V13m,Id,Hazard))

test3  = subset(dtest,select=c(T1_V1,T1_V2,T1_V3,T1_V10,T1_V13,T1_V14,T2_V1,T2_V2,T2_V4,T2_V6,T2_V7,T2_V9,T2_V10,
                               T2_V14,T2_V15,T1_V4m,T1_V5m,T1_V7m,T1_V8m,T1_V9m,T1_V11m,T1_V12m,T1_V15m,T1_V16m,
                               T2_V5m,T2_V13m,Id))

# Set parameters for the model

seed=12; nround = 20000; print_n = 200; early_stop = 200
param  = list("objective" = "reg:linear",
              "eta" = 0.01,
              "min_child_weight" = 5, 
              "subsample" = .8, 
              "colsample_bytree" = .8, 
              "scale_pos_weight" = 1.0,
              "max_depth" = 9,
              "silent" = 1,"nthread" = 2)

cat('MODEL 3. Building...\n')
print(param)
call3=xgb_pred(train=train3,test=test3,seed=seed,
               param=param,nround=nround,print_n=print_n,early_stop=early_stop,
               feval=evalgini,maximize = T)
pred3=call3$pred

# MODEL 4.

train.data = subset(train.data, select=-c(T1_V10,T1_V13,T2_V7,T2_V10))
test.data  = subset(test.data,  select=-c(T1_V10,T1_V13,T2_V7,T2_V10))

fv = NULL
for(i in 1:ncol(train.data)) fv [i] = length(levels(factor(train.data[,i])))
names(fv)=names(train.data)
print(fv)

for (i in 3:ncol(train.data)) {
        train.data[,i]=(as.integer(
                mapvalues(train.data[,i],levels(as.factor(train.data[,i])),
                          1:length(levels(as.factor(train.data[,i])))))-1)/(fv1[i]-1)*100
}
for (i in 2:ncol(test.data)) {
        test.data[,i]=(as.integer(
                mapvalues(test.data[,i],levels(as.factor(test.data[,i])),
                          1:length(levels(as.factor(test.data[,i])))))-1)/(fv1[i+1]-1)*100
}
        
dtrain = train.data
dtest  = test.data
set.seed(350)
inTrain = createDataPartition(y=dtrain$Hazard,p=.7,list=F)
trn1 = dtrain[ inTrain,]
ts   = dtrain[-inTrain,]

set.seed(350)
inTrain1 = createDataPartition(y=trn1$Hazard,p=.887,list=F)
trn = trn1[ inTrain1,]
cv  = trn1[-inTrain1,]       
label = as.numeric(trn[[2]])

data = as.matrix(trn[c(3:ncol(trn))])
data.cv = as.matrix(cv[c(3:ncol(cv))])
data.ts = as.matrix(ts[c(3:ncol(ts))])
data.test = as.matrix(dtest[c(2:ncol(dtest))]) 
        
label.cv = as.numeric(cv[[2]])
label.ts = as.numeric(ts[[2]])

data=rbind(data,data.ts)
label=c(label,label.ts)

xgmat = xgb.DMatrix(data, label = label)
        
param = list("objective" = "reg:linear",
        "eta" = 0.01,
        "min_child_weight" = 5, 
        "subsample" = .8, 
        "colsample_bytree" = .85, 
        "scale_pos_weight" = .87,
        "max_depth" = 6,
        "silent" = 1,
        "nthread" = 2)
              
xgval = xgb.DMatrix(data=data.cv,label=label.cv)
xgtrain = xgb.DMatrix(data=data,label=label)
watchlist = list(val=xgval, train=xgtrain)
nround = 35000

cat('MODEL 4. Building...\n')

set.seed(350)
bstgini = xgb.train(param, xgmat, nround, watchlist,feval = evalgini, print.every.n = 100,early.stop.round = 200, maximize = T);
print(bstgini$bestScore)

pred4 = predict(bstgini,xgb.DMatrix(data.test),ntreelimit=bstgini$bestInd)

# Build final ensemble 

test.pred = .406 * (log(pred1) ** .0378) + 
            .47  * (pred2 ** .04) + 
            .15  * (pred3 ** .04) +
            .37  * (pred4 ** .043)

Submission=data.frame("Id"=dtest$Id,"Hazard" = test.pred)
write.csv(Submission,"./predictions.csv",row.names=FALSE,quote=FALSE)

# End of story
End.time=Sys.time()
print(End.time-Start.time)
