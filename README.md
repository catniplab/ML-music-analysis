# ML-music-analysis
How do machines learn to predict music?

## Intended Workflow

We will train just-in-time `jit` compiled pytorch models on four music datasets and store the results using `sacred`.

MATLAB files for the music datasets can be found in `data` with the license from locuslab, who compiled them.

In `src`, `load_data` allows for the construction of a data loader for a specific dataset. `base_models` is where we define our custom pytorch models. `models` implements various things on top of our base models: linear readout, custom initialization, gradient clipping. `experiment.py` defines the training loop and default configs.

All tensors representing music should be indexed via `[sample, time, note]`. Iteration over a DataLoader involves a mask for each batch which is used for accuracte computation of loss and accuracy.

Run an experiment with `python executable.py`, configurations for the experiment can be found in the script and different configuration options are roughly documented in `src/experiment.py`. `query_results.py` can be run to find out which directories in `results` contain results for experiments with specified configurations. `plotting.py` contains a bunch of helper functions for plotting training curves and information about hidden weights.