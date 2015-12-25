#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)

#--
N_CLUSTERS = 32

# N_TREE = 100
N_TREE = as.numeric(args[1])
FOLDER = paste0('saved_runs/auc_sample_size/nt_', N_TREE, '/')
print(FOLDER)
dir.create(FOLDER, showWarnings = FALSE)

N_ITERATION = 50
#--
library(plyr)
require(caret)
require(doParallel)
require(foreach)
#--
df=read.csv('../../data/FSS.csv',row.names=1)
df=subset(df,sex==1)
df$control[df$control==0] ="ASD"
df$control[df$control==1] ="TDC"
df$control = as.factor(df$control)

intensity_columns = grep("_intensity",names(df))
df = df[-intensity_columns]

print(dim(df))



# normalize volume features by TIV
d=df[10:length(df)]
col_vol=grep("_volume",names(d))  
TIV=df$EstimatedTotalIntraCranialVol_volume
d[col_vol]=as.data.frame(sweep(data.matrix(d[col_vol]),1,TIV,'/'))
d=d[-length(d)]

ASD = d[df$control == "ASD", ]
TDC = d[df$control == "TDC", ] 

## setup parallel backend to use multiple processors
cl<-makeCluster(N_CLUSTERS)
registerDoParallel(cl)
    
fitControl <- trainControl(method = "repeatedcv",
                             number = 5,
                             repeats = 1,
                             ## Estimate class probabilities
                             classProbs = TRUE,
                             ## Evaluate performance using 
                             ## the following function
                             summaryFunction = twoClassSummary)
# mtry
Mtry=round(sqrt(ncol(d)))
grid <-  expand.grid(mtry=seq(Mtry-10 ,Mtry+10, 5))      
print(grid)
df_AUC = data.frame(AUC_mean=NA, AUC_sd=NA)
for (sample_size in c(seq(20, 734, 10), 100)){
  strt=Sys.time()

  print(paste0("sample size = ", sample_size))
  collect_iterations = list()
  collect_AUC = list()
  for ( it in 1:N_ITERATION){
    # print(paste0("it=", it))
    TrainASD = ASD[sample(nrow(ASD), round(sample_size/2)), ] # randomly sample the rows
    TrainTDC = TDC[sample(nrow(TDC), round(sample_size/2)), ] # randomly sample the rows
    Train = rbind(TrainASD, TrainTDC)

    yTrain = df[rownames(Train), "control"]
    
    # set.seed(123)
    fit <- train(x=Train, y=yTrain,
                 method = "rf",
                 trControl = fitControl,
                 tuneGrid = grid,
                 verbose = FALSE,
                 metric = "ROC",
                 ntree = N_TREE)
    # print(fit)

    collect_iterations = c(collect_iterations, fit)
    AUC = max(fit$results["ROC"])
    collect_AUC = c(collect_AUC, AUC)
  }
  print(paste0("Finished in ", Sys.time()-strt))
  
  save(collect_iterations,file=paste0(FOLDER ,'fit_ ', sample_size,'.Rdata'))
  print(paste0("AUC=",mean(AUC)))
  df_AUC = rbind(df_AUC, data.frame(AUC_mean = mean(unlist(collect_AUC)), AUC_sd = sd(unlist(collect_AUC))))
}
stopCluster(cl)
  