# ML-music-analysis
How do machines learn to predict music?

## Intended Workflow

We will train just-in-time `jit` compiled pytorch models on four music datasets and store the results using `sacred`.

MATLAB files for the music datasets can be found in `data` with the license from locuslab, who compiled them.

In `src`, `load_data` allows for the construction of a data loader for a specific dataset. `base_models` is where we define our custom pytorch models. `models` implements various things on top of our base models: linear readout, custom initialization, gradient clipping. `experiment.py` is where we will define the training loop and default configs.

Note that all tensors representing music should be indexed via `[sample, time, note]`.

Run the current default experiment by `python executable.py`.