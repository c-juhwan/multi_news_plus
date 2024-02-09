# from starting_idx to ending_idx, run the job
clear
STARTING_IDX=0
ENDING_IDX=56200
INCREMENTAL=100

for i in $(seq $STARTING_IDX $INCREMENTAL $ENDING_IDX)
do
    # echo "Starting from $i to $(($i + $INCREMENTAL))"
    python cleansing.py --start_idx=$i --end_idx=$(($i + $INCREMENTAL))
done
