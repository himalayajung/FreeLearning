#! /bin/sh
export DATAFILE=FSS.csv
# export DATAFILE=zernike_moments_native.csv
#export CLASSIFICATION=RF
export ATTRIBUTE=ALL
export VALIDATION=skf
export NFOLD=10
#export SEARCH=random
export N_ITER_RANDOM=20
#export SCORING=roc_auc
# export SCORING=roc_auc_weighted

# export SCORING=accuracy
export SEX=1

export CLASSIFICATION=$1
export SCORING=$2
export SEARCH=$3
export NTREE=$4
export REMOVE_CORR=$5

export SUBSETTING=$6
export BALANCE_CLASSES=$7 # no  # OR random upsample SMOTE
export SAVE_FOLDER=$8
export RUN_VARIABLE=$9
# export MATCH_TYPE=$10 # bash cannot accept more than 9 arguments


export SAVE_FOLDER_PATH=/fs-research01/niagroup/gjkatuwal/codes/python/freesurfer_classification/saved_runs/$SAVE_FOLDER
echo $SAVE_FOLDER_PATH
mkdir -p $SAVE_FOLDER_PATH

export ADOS1=(0 9.99)
export ADOS2=(10 14)
export ADOS3=(14.01 30)

# export ADOS1=(0 11)
# export ADOS2=(11.01 30)

export AGE1=(0 12.99)
export AGE2=(13 17.99)
export AGE3=(18 40)

# export AGE2=(13 24.99)
# export AGE3=(25 40)
# export VIQ1=(0 99.99)
# export VIQ2=(100 115)
# export VIQ3=(115.01 180)

# export VIQ1=(0 90)
# export VIQ2=(90.01 114.99)
# # export VIQ2=(85.01 114.99)
# export VIQ3=(115 180)

export VIQ1=(75 90)
export VIQ2=(90.01 114.99)
# export VIQ2=(85.01 114.99)
export VIQ3=(115 150)

# export VIQ1=(0 85)
# export VIQ2=(85.01 114.99)
# export VIQ3=(115 180)

export AS1=(1 5)
export AS2=(5.01 7)
export AS3=(7.01 10)

if [ $SUBSETTING -eq 1 ]
	then
	printf "\n Individual Subsetting ..... upsample"

	if [ $RUN_VARIABLE -eq 1 ]
		then
		printf "\n Subsetting based on ADOS ....."
		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -bc $BALANCE_CLASSES -vd $VALIDATION -nf $NFOLD -sex $SEX -ADOS ${ADOS1[*]} -stats 
		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -bc $BALANCE_CLASSES -vd $VALIDATION -nf $NFOLD -sex $SEX -ADOS ${ADOS2[*]} -stats 
		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -bc $BALANCE_CLASSES -vd $VALIDATION -nf $NFOLD -sex $SEX -ADOS ${ADOS3[*]} -stats 
	elif [ $RUN_VARIABLE -eq 2 ]
		then
		printf "\n Subsetting based on age ....."
		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -bc $BALANCE_CLASSES -vd $VALIDATION -nf $NFOLD -sex $SEX -age ${AGE1[*]} -stats 
		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -bc $BALANCE_CLASSES -vd $VALIDATION -nf $NFOLD -sex $SEX -age ${AGE2[*]} -stats 
		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -bc $BALANCE_CLASSES -vd $VALIDATION -nf $NFOLD -sex $SEX -age ${AGE3[*]} -stats 
	elif [ $RUN_VARIABLE -eq 3 ]
		then
		printf "\n Subsetting based on VIQ ....."
		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -bc $BALANCE_CLASSES -vd $VALIDATION -nf $NFOLD -sex $SEX -VIQ ${VIQ1[*]} -stats 
		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -bc $BALANCE_CLASSES -vd $VALIDATION -nf $NFOLD -sex $SEX -VIQ ${VIQ2[*]} -stats 
		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -bc $BALANCE_CLASSES -vd $VALIDATION -nf $NFOLD -sex $SEX -VIQ ${VIQ3[*]} -stats 

	elif [ $RUN_VARIABLE -eq 4 ]
		then
		printf "\n Subsetting based on AS ....."
		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -bc $BALANCE_CLASSES -vd $VALIDATION -nf $NFOLD -sex $SEX -AS ${AS1[*]} -stats 
		 python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -bc $BALANCE_CLASSES -vd $VALIDATION -nf $NFOLD -sex $SEX -AS ${AS2[*]} -stats 
		 python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -bc $BALANCE_CLASSES -vd $VALIDATION -nf $NFOLD -sex $SEX -AS ${AS3[*]} -stats 

		 python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -bc $BALANCE_CLASSES -vd $VALIDATION -nf $NFOLD -sex $SEX -AS 4 5 -stats 

		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -bc $BALANCE_CLASSES -vd $VALIDATION -nf $NFOLD -sex $SEX -AS 6 6 -stats 
		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -bc $BALANCE_CLASSES -vd $VALIDATION -nf $NFOLD -sex $SEX -AS 7 7 -stats 
		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -bc $BALANCE_CLASSES -vd $VALIDATION -nf $NFOLD -sex $SEX -AS 8 8 -stats 
		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -bc $BALANCE_CLASSES -vd $VALIDATION -nf $NFOLD -sex $SEX -AS 9 9 -stats 
		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -bc $BALANCE_CLASSES -vd $VALIDATION -nf $NFOLD -sex $SEX -AS 10 10 -stats 
		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -bc $BALANCE_CLASSES -vd $VALIDATION -nf $NFOLD -sex $SEX -AS 0 10 -stats 
	else
		printf "\n Options 1:ADOS, 2:age, 3:VIQ"
	fi

elif [ $SUBSETTING -eq 2 ]
	then
	printf "\n Individual Subsetting with matched subjects....."

	if [ $RUN_VARIABLE -eq 1 ]
		then
		printf "\n Subsetting based on ADOS ....."
		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -vd $VALIDATION -nf $NFOLD -sex $SEX -ADOS ${ADOS1[*]} -mg 110 -mt $MATCH_TYPE -stats  
		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -vd $VALIDATION -nf $NFOLD -sex $SEX -ADOS ${ADOS2[*]} -mg 1014 -mt $MATCH_TYPE -stats  
		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -vd $VALIDATION -nf $NFOLD -sex $SEX -ADOS ${ADOS3[*]} -mg 1425 -mt $MATCH_TYPE -stats  

		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -vd $VALIDATION -nf $NFOLD -sex $SEX -ADOS 6 9.99 -mg 610 -mt $MATCH_TYPE -stats  
	 	python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -vd $VALIDATION -nf $NFOLD -sex $SEX -ADOS 14.01 20 -mg 1420 -mt $MATCH_TYPE -stats  
		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -vd $VALIDATION -nf $NFOLD -sex $SEX -ADOS 0 25 -mg 0 -mt $MATCH_TYPE -stats  
	elif [ $RUN_VARIABLE -eq 2 ]
		then
		printf "\n Subsetting based on age ....."
		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -vd $VALIDATION -nf $NFOLD -sex $SEX -age ${AGE1[*]} -mg 1 -mt $MATCH_TYPE -stats  
		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER-vd $VALIDATION  -nf $NFOLD -sex $SEX -age ${AGE2[*]} -mg 2 -mt $MATCH_TYPE -stats  
		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -vd $VALIDATION -nf $NFOLD -sex $SEX -age ${AGE3[*]} -mg 3 -mt $MATCH_TYPE -stats  
	elif [ $RUN_VARIABLE -eq 3 ]
		then
		printf "\n Subsetting based on VIQ ....."
		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -vd $VALIDATION -nf $NFOLD -sex $SEX -VIQ ${VIQ1[*]} -mg 1 -mt 1 -stats  
		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -vd $VALIDATION -nf $NFOLD -sex $SEX -VIQ ${VIQ2[*]} -mg 2 -mt 1 -stats  
		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -vd $VALIDATION -nf $NFOLD -sex $SEX -VIQ ${VIQ3[*]} -mg 3 -mt 1 -stats  

	elif [ $RUN_VARIABLE -eq 4 ]
		then
		printf "\n Subsetting based on AS ....."
		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -vd $VALIDATION -nf $NFOLD -sex $SEX -AS ${AS1[*]} -mg 15 -mt 2 -stats  
		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -vd $VALIDATION -nf $NFOLD -sex $SEX -AS ${AS2[*]} -mg 67 -mt 2 -stats  
		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -vd $VALIDATION -nf $NFOLD -sex $SEX -AS ${AS3[*]} -mg 810 -mt 2 -stats  


		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -vd $VALIDATION -nf $NFOLD -sex $SEX -AS 4 5 -mg 45 -mt 2 -stats  

		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -vd $VALIDATION -nf $NFOLD -sex $SEX -AS 6 6 -mg 6 -mt 2 -stats  
	 	python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -vd $VALIDATION -nf $NFOLD -sex $SEX -AS 7 7 -mg 7 -mt 2 -stats  
		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -vd $VALIDATION -nf $NFOLD -sex $SEX -AS 8 8 -mg 8 -mt 2 -stats  
		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -vd $VALIDATION -nf $NFOLD -sex $SEX -AS 9 9 -mg 9 -mt 2 -stats  
		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -vd $VALIDATION -nf $NFOLD -sex $SEX -AS 10 10 -mg 10 -mt 2 -stats 
		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -vd $VALIDATION -nf $NFOLD -sex $SEX -AS 0 10 -mg 0 -mt 2 -stats 

	else
		printf "\n Options 1:ADOS, 2:age, 3:VIQ"
	fi

elif [ $SUBSETTING -eq 3 ]
	then
	printf "\n 3-way Subsetting ....."
	if [ $RUN_VARIABLE -eq 1 ]
		then
		printf ${ADOS1[*]}
		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -bc $BALANCE_CLASSES -vd $VALIDATION -nf $NFOLD -sex $SEX -ADOS ${ADOS1[*]} -age ${AGE1[*]} -VIQ ${VIQ1[*]} -stats 
		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -bc $BALANCE_CLASSES -vd $VALIDATION -nf $NFOLD -sex $SEX -ADOS ${ADOS1[*]} -age ${AGE1[*]} -VIQ ${VIQ2[*]} -stats 
		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -bc $BALANCE_CLASSES -vd $VALIDATION -nf $NFOLD -sex $SEX -ADOS ${ADOS1[*]} -age ${AGE1[*]} -VIQ ${VIQ3[*]} -stats 

		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -bc $BALANCE_CLASSES -vd $VALIDATION -nf $NFOLD -sex $SEX -ADOS ${ADOS1[*]} -age ${AGE2[*]} -VIQ ${VIQ1[*]} -stats 
		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -bc $BALANCE_CLASSES -vd $VALIDATION -nf $NFOLD -sex $SEX -ADOS ${ADOS1[*]} -age ${AGE2[*]} -VIQ ${VIQ2[*]} -stats 
		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -bc $BALANCE_CLASSES -vd $VALIDATION -nf $NFOLD -sex $SEX -ADOS ${ADOS1[*]} -age ${AGE2[*]} -VIQ ${VIQ3[*]} -stats 

		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -bc $BALANCE_CLASSES -vd $VALIDATION -nf $NFOLD -sex $SEX -ADOS ${ADOS1[*]} -age ${AGE3[*]} -VIQ ${VIQ1[*]} -stats 
		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -bc $BALANCE_CLASSES -vd $VALIDATION -nf $NFOLD -sex $SEX -ADOS ${ADOS1[*]} -age ${AGE3[*]} -VIQ ${VIQ2[*]} -stats 
		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -bc $BALANCE_CLASSES -vd $VALIDATION -nf $NFOLD -sex $SEX -ADOS ${ADOS1[*]} -age ${AGE3[*]} -VIQ ${VIQ3[*]} -stats 
	
	elif [ $RUN_VARIABLE -eq 2 ]
		then
		printf ${ADOS2[*]}
		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -bc $BALANCE_CLASSES -vd $VALIDATION -nf $NFOLD -sex $SEX -ADOS ${ADOS2[*]} -age ${AGE1[*]} -VIQ ${VIQ1[*]} -stats 
		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -bc $BALANCE_CLASSES -vd $VALIDATION -nf $NFOLD -sex $SEX -ADOS ${ADOS2[*]} -age ${AGE1[*]} -VIQ ${VIQ2[*]} -stats 
		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -bc $BALANCE_CLASSES -vd $VALIDATION -nf $NFOLD -sex $SEX -ADOS ${ADOS2[*]} -age ${AGE1[*]} -VIQ ${VIQ3[*]} -stats   

		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -bc $BALANCE_CLASSES -vd $VALIDATION -nf $NFOLD -sex $SEX -ADOS ${ADOS2[*]} -age ${AGE2[*]} -VIQ ${VIQ1[*]} -stats 
		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -bc $BALANCE_CLASSES -vd $VALIDATION -nf $NFOLD -sex $SEX -ADOS ${ADOS2[*]} -age ${AGE2[*]} -VIQ ${VIQ2[*]} -stats 
		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -bc $BALANCE_CLASSES -vd $VALIDATION -nf $NFOLD -sex $SEX -ADOS ${ADOS2[*]} -age ${AGE2[*]} -VIQ ${VIQ3[*]} -stats 

		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -bc $BALANCE_CLASSES -vd $VALIDATION -nf $NFOLD -sex $SEX -ADOS ${ADOS2[*]} -age ${AGE3[*]} -VIQ ${VIQ1[*]} -stats 
		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -bc $BALANCE_CLASSES -vd $VALIDATION -nf $NFOLD -sex $SEX -ADOS ${ADOS2[*]} -age ${AGE3[*]} -VIQ ${VIQ2[*]} -stats 
		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -bc $BALANCE_CLASSES -vd $VALIDATION -nf $NFOLD -sex $SEX -ADOS ${ADOS2[*]} -age ${AGE3[*]} -VIQ ${VIQ3[*]} -stats 
	
	elif [ $RUN_VARIABLE -eq 3 ]
		then
		printf ${ADOS3[*]}
		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -bc $BALANCE_CLASSES -vd $VALIDATION -nf $NFOLD -sex $SEX -ADOS ${ADOS3[*]} -age ${AGE1[*]} -VIQ ${VIQ1[*]} -stats 
		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -bc $BALANCE_CLASSES -vd $VALIDATION -nf $NFOLD -sex $SEX -ADOS ${ADOS3[*]} -age ${AGE1[*]} -VIQ ${VIQ2[*]} -stats 
		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -bc $BALANCE_CLASSES -vd $VALIDATION -nf $NFOLD -sex $SEX -ADOS ${ADOS3[*]} -age ${AGE1[*]} -VIQ ${VIQ3[*]} -stats   

		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -bc $BALANCE_CLASSES -vd $VALIDATION -nf $NFOLD -sex $SEX -ADOS ${ADOS3[*]} -age ${AGE2[*]} -VIQ ${VIQ1[*]} -stats 
		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -bc $BALANCE_CLASSES -vd $VALIDATION -nf $NFOLD -sex $SEX -ADOS ${ADOS3[*]} -age ${AGE2[*]} -VIQ ${VIQ2[*]} -stats 
		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -bc $BALANCE_CLASSES -vd $VALIDATION -nf $NFOLD -sex $SEX -ADOS ${ADOS3[*]} -age ${AGE2[*]} -VIQ ${VIQ3[*]} -stats 

		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -bc $BALANCE_CLASSES -vd $VALIDATION -nf $NFOLD -sex $SEX -ADOS ${ADOS3[*]} -age ${AGE3[*]} -VIQ ${VIQ1[*]} -stats 
		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -bc $BALANCE_CLASSES -vd $VALIDATION -nf $NFOLD -sex $SEX -ADOS ${ADOS3[*]} -age ${AGE3[*]} -VIQ ${VIQ2[*]} -stats 
		python pipeline.py $DATAFILE -c $CLASSIFICATION -a $ATTRIBUTE -nt $NTREE -remove_correlated $REMOVE_CORR --search $SEARCH -n_iter $N_ITER_RANDOM --scoring $SCORING -folder $SAVE_FOLDER -bc $BALANCE_CLASSES -vd $VALIDATION -nf $NFOLD -sex $SEX -ADOS ${ADOS3[*]} -age ${AGE3[*]} -VIQ ${VIQ3[*]} -stats 
	else
		printf "\n Options 1:ADOS1, 2:ADOS2, 3:ADOS3"
	fi
else
	printf "\n Subsetting options are 1: individual, 2: 3-way"
fi

		















