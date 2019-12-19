RAW_DATA_DIR=data/LJSpeech-1.1
DATA_DIR=data/lj

PREPROCESS="python3 preprocess_synthesizer_data.py"

echo "Downloading and preprocessing data..."
${PREPROCESS} --raw-data-dir ${RAW_DATA_DIR} --data-dir ${DATA_DIR}

echo "Preprocessing data"
${PREPROCESS} --data-dir ${DATA_DIR}
