# ML-music-analysis
How do machines learn to predict music?

## Intended Workflow

We will use `pytorch-ignite` to train just-in-time `jit` compiled pytorch models on four music datasets and store the results using `sacred`.

MATLAB files for the music datasets can be found in `data` with the license from locuslab, who compiled them.

In `src`, `load_data` allows for the construction of a data loader for a specific dataset. `custom_models` is where we define our custom pytorch models. `jits_model` allows for the construction of a just-in-time compiled model with a linear readout layer. `experiment.py` is where we will define the training loop and default configs.

Note that all tensors representing music should be indexed via `[sample, time, note]`.