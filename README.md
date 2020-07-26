# ML-music-analysis
How do machines learn to predict music?

## Intended Workflow

We will train just-in-time `jit` compiled pytorch models on four music datasets and store the results using `sacred`.

MATLAB files for the music datasets can be found in `data` with the license from locuslab, who compiled them.

In `src`, we have code for training `sklearn` regression models and `pytorch` neural networks. `experiment.py` defines how these two libraries are called and how the results are stored. in `src/neural_nets`, `load_data.py` defines a function which gets a `pytorch` `DataLoader` for a specified dataset, including masking to account for differing sequences lengths. `base_models.py` is for defining new `pytorch` modules specifying the evolution of some hidden state., `models.py` defines readout for these models and adds things like gradient clipping, while `initialization.py` is for appropriately initializing different types of models. `metrics.py` defines accuracy and loss specific to the polyphonic music prediction task.

All tensors representing music should be indexed via `[sample, time, note]`. Iteration over a DataLoader involves a mask for each batch which is used for accuracte computation of loss and accuracy.

Run an experiment with `python executable.py`, configurations for the experiment can be found in the script and different configuration options are roughly documented in `src/experiment.py`. `query_results.py` can be used to find out which directories in `results` contain results for experiments with specified configurations. `plotting.py` contains a bunch of helper functions for plotting training curves and information about hidden weights.
