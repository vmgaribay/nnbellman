#!/bin/bash
#SBATCH --job-name=SchB
#SBATCH -p gpu
#SBATCH --gpus=1

#              d-hh:mm:ss
#SBATCH --time=02:30:00


module load 2023

# Environment (Snellius specific)
source /home/vgaribay/anaconda3/etc/profile.d/conda.sh
#conda env create -f ../environment.yml --name dgl_ptm_gpu
conda activate NNFun

echo "Script: PudgeSixSchemeB.sh"
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
                        variation="-c SchemeB.json --a $a --lr $lr --bs $bs --s $s --mn $mn --n $name"
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

