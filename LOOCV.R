#!/usr/bin/env Rscript

# parallel random forest under LOOCV framework
# code for the Experiment 1 of the IEEE EMBS 2015 paper
# @author Gajendra Jung Katuwal

# example
# run in commandline as:
# ./LOOCV.R 10 15 32 
# run RF in LOOCV framework, use mtry = 10 to 15 for grid tuning, initialize 32 clusters
args <- commandArgs(trailingOnly = TRUE)
require(randomForest)
require(doParallel)
require(foreach)

mtry1 = args[1]
mtry2 = args[2]
N_CLUSTER = as.numeric(args[3])
if (is.na(N_CLUSTER)){
  N_CLUSTER = 24
}

print(mtry1)
print(mtry2)
print(N_CLUSTER)



## Settings
set.seed(123)
include_phenotype=FALSE
remove_intensity = TRUE
only_male=1
scale_by_TIV=1


# Loop along the features
feature="_all"

# Data 
df=read.csv('../../data/FS_imputed.csv',row.names=1)

df$control=as.factor(df$control)
if(only_male==1){
  male="male_"
  print('only male subjects used')
  df = df[df$sex==1,]
} else{ 
  male=""
}
phenotype = df[c("age", "site", "VIQ", "PIQ", "FIQ")]


d=df[11:length(df)] # subsetting morphometric features

if (remove_intensity){
  print("Intensity features removed")
  intensity_columns = grep("_intensity",names(d))
  d = d[-intensity_columns]
}

if(include_phenotype==1){
  print('phenotype info included')
  pheno="pheno_"
  d = cbind(d, phenotype)
 } else{ 
        pheno=""
      } 


filename=paste0('saved_runs/experiment1/DB_effect/RF_', pheno)
logfile = paste0(filename,'.log')
con = file(logfile)
sink(con, append=TRUE, split = TRUE)
# sink(con, append=TRUE, type="message")

if (feature=="_all" && scale_by_TIV==1) {
          col_vol=grep("_volume",names(d))  
          TIV=df$EstimatedTotalIntraCranialVol_volume
          d[col_vol]=as.data.frame(sweep(data.matrix(d[col_vol]),1,TIV,'/'))
          d=d[-length(d)]
        }

print(dim(d))
## setup parallel backend to use multiple processors
n_cluster = N_CLUSTER
cl<-makeCluster(n_cluster)
registerDoParallel(cl)

for (mtry in mtry1:mtry2){
  print(paste0("mtry = ", mtry))
    system.time({
    cv.model <- foreach(i = 1:nrow(d), .packages = c("randomForest")) %dopar% {
      Train=d[-i,]
      Test=d[i,]
      yTrain=df$control[-i]
      yTest=df$control[i]

    randomForest(x=Train, y=yTrain, xtest=Test, ytest=yTest, n.trees=5000, mtry=mtry, importance=TRUE, do.trace=TRUE)
    }
  })
savefile = paste0(filename,'_', mtry, '.Rdata')
print(savefile)
save(cv.model,file=savefile)


}    
stopCluster(cl)

 
