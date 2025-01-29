#!/bin/bash
#SBATCH --job-name=SchA
#SBATCH -p gpu
#SBATCH --gpus=1

#              d-hh:mm:ss
#SBATCH --time=02:30:00


module load 2023
#python -m venv NNFun 
source NNFun/bin/activate
#pip install -r requirements.txt


echo "Script: PudgeSixSchemeC.sh"
# Model Training for Experiments

architecture=("PudgeSixLayer")
learningrate=("0.001")
batchsize=("64")
maxnodes=("2048")
seed=("21" "42" "84")

total_runs=$(( ${#architecture[@]} * ${#learningrate[@]} * ${#batchsize[@]} * ${#maxnodes[@]} * ${#seed[@]} ))
counter=0
restart=0
earlystop=3

# Loop through every combination
for a in "${architecture[@]}"
do
    for lr in "${learningrate[@]}"
    do
        for bs in "${batchsize[@]}"
        do
            for mn in "${maxnodes[@]}"
            do
                for s in "${seed[@]}"
                do
                    ((counter++))
                    if [ "$counter" -ge "$restart" ] && [ "$counter" -le "$earlystop" ]; then
                        name="both_${a}_${mn}_${s}" 
                        date=$(date)
                        echo "$date Started run $counter/$total_runs with architecture: $a, learning rate: $lr, batch size: $bs, max nodes: $mn, and seed: $s"
                        variation="-c SchemeC.json --a $a --lr $lr --bs $bs --s $s --mn $mn --n $name"
                        python train.py $variation  
                        date=$(date)
                        echo "$date Finished run $counter/$total_runs with architecture: $a, learning rate: $lr, batch size: $bs, max nodes: $mn, and seed: $s"
                    fi
                done    
            done
            
        done
    done
done

wait


echo " $date - Runs $restart through $earlystop are attempted for seeds 21, 42, and 84."

