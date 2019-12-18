#echo "Downloading requirements"
#pip install -r requirements.txt

RAW_DATA_DIR=data/LJSpeech-1.1
DATA_DIR=data/lj
MODELS_DIR=models
SMALL=${DATA_DIR}/small.pickle

PREPROCESS="python3 preprocess_synthesizer_data.py"

if [ ! -f "${DATA_DIR}/small.pickle" ]; then

    if [ ! -f "${DATA_DIR}/all.raw.pickle" ]; then

        echo "Downloading and preprocessing data..."
        ${PREPROCESS} --raw-data-dir ${RAW_DATA_DIR} --data-dir ${DATA_DIR}

    else

        echo "Preprocessing data"
        ${PREPROCESS} --data-dir ${DATA_DIR}

    fi

fi

#TRAINER="python trainer.py"

#echo "Training..."
#${TRAINER} \
#        --train_data ${SMALL} \
#        --test_data ${SMALL} \
#        --num_epochs 1 \
#        --batch_size 32 \
#        --loss "binary_crossentropy" \
#        --filters 512 \
#        --optimizer "adam" \
#        --learning_rate 0.001 \

#echo "done"

# eof
