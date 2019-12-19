## Overview


### Dependencies:

- tensorflow==1.10.0
- librosa==0.7.1
- keras==2.2.4
- nnmnkwii
- python version 3.6

### Steps to train and validate Tacotron: 
1. Download LJ dataset
2. Create directory `/data` and untar LJ dataset in there
3. Create directory `/data/lj` (for the compressed dataset)
4. Create directory `/results`
5. run `./run_synthesizer_expt.sh`
6. `python3 train.py`
7. `python3 validate_test.py` (to validate reconstruction)
