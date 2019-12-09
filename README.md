## Overview

There's currently a toy dataset in `data/lj/small.pickle` with about 20 examples (20 wav files as numpy arrays and the corresponding sentences). It's small enough to use for debugging purposes, especially since the entire dataset is around 3.5 GB (which I have locally).

To run the entire experiment so far, run `./run_synthesizer_expt.sh`. You might have some pathway issues with the raw-dataset-directory (since I have it pointed to my local directory, but if that's the case, just comment that line out). Since I have a check for the small.pickle file, it should skip right to the model file `trainer.py` and start running from there.

