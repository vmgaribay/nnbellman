#!/bin/bash
#SBATCH --job-name=csNNS84
#SBATCH -p gpu
#SBATCH --gpus=1

#              d-hh:mm:ss
#SBATCH --time=12:30:00


module load 2022
#python -m venv NNFun 
source NNFun/bin/activate
#pip install -r requirements.txt

#cp $HOME/NNFunction/Trial3/DatasetSmall/ResultsFinal.csv "$TMPDIR"
#cp $HOME/NNFunction/Trial3/DatasetSmall/AgentData.csv "$TMPDIR"
#mkdir $TMPDIR/saved/i_a


# Experimental setup for seed 84

architecture=("ThreeLayer" "FourLayer" "FiveLayer" "PudgeFiveLayer" "PudgeSixLayer")
learningrate=("0.001" "0.01" "0.0001")
batchsize=("64" "128" "512")
maxnodes=("512" "1024" "2048")

total_runs=$(( ${#architecture[@]} * ${#learningrate[@]} * ${#batchsize[@]} * ${#maxnodes[@]} ))
counter=0
restart=0
earlystop=135

echo "Script: cs_both_grid_search_S84gpu.sh"
# Loop through every combination
for a in "${architecture[@]}"
do
    for lr in "${learningrate[@]}"
    do
        for bs in "${batchsize[@]}"
        do
            for mn in "${maxnodes[@]}"
            do
                ((counter++))
                if [ "$counter" -ge "$restart" ] && [ "$counter" -le "$earlystop" ]; then
                    name="both_${a}_${mn}" 
                    date=$(date)
                    echo "$date Started run $counter/$total_runs with architecture: $a, learning rate: $lr, batch size: $bs, max nodes: $mn"
                    variation="-c both_config_snell3_cons_scaled.json --a $a --lr $lr --bs $bs --s 84 --mn $mn --n $name"
                    python train.py $variation 
                    date=$(date) 
                    echo "$date Finished run $counter/$total_runs with architecture: $a, learning rate: $lr, batch size: $bs, max nodes: $mn"
                fi
            done
            
        done
    done
done

wait

#cp -r "$TMPDIR"/saved $HOME/NNFunction/Trial3/saved

echo " $date - Runs $restart through $earlystop are complete for seed 84."

