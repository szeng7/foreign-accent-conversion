# run from foreign-accent-conversion
# exit any venvs
. venv3.6/bin/activate
python3 super_cool_looking_gui.py &&
FILE_PATH=`cat file_path.txt`
deactivate

. ../deepspeech/deepspeech-venv/bin/activate
deepspeech --model ../deepspeech/deepspeech-0.6.0-models/output_graph.pbmm --lm ../deepspeech/deepspeech-0.6.0-models/lm.binary --trie ../deepspeech/deepspeech-0.6.0-models/trie --audio wav_src_path --model ../deepspeech/deepspeech-0.6.0-models/output_graph.pbmm --lm ../deepspeech/deepspeech-0.6.0-models/lm.binary --trie ../deepspeech/deepspeech-0.6.0-models/trie --audio $FILE_PATH >out.txt
deactivate

. venv3.6/bin/activate
python synthesize.py
deactivate

