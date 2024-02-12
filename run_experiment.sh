clear

DEVICE=cuda:0
MODEL_LIST=(bart t5)
TASK_LIST=(multi_news_plus multi_news)

for MODEL in ${MODEL_LIST[@]}; do
    for DATASET in ${TASK_LIST[@]}; do
        python main.py --task=summarization --job=preprocessing \
                       --task_dataset=${DATASET} --model_type=${MODEL}

        python main.py --task=summarization --job=training \
                       --task_dataset=${DATASET} --model_type=${MODEL} --device=${DEVICE}

        python main.py --task=summarization --job=testing \
                       --task_dataset=${DATASET} --model_type=${MODEL} --device=${DEVICE}
    done
done