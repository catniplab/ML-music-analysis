"""
This script is for generating new music based on the LocusLab datasets and the models trained on them.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchsso.optim as soptim
import torch.nn.functional as F
import random

from torch.utils.data import DataLoader
from scipy.io import loadmat
from sacred import Experiment
from torch import Tensor, device
from copy import deepcopy
from time import sleep
from tqdm import tqdm

from src.neural_nets.models import get_model
from src.midi.utils import to_midi, make_music


# create a new sacred experiment whose name is an integer
ex = Experiment(name=str(random.randint(0, 1000000)))


# default configurations
@ex.config
def cfg():

    # supported datasets
    # JSB_Chorales (short)
    # Nottingham (medium)
    # Piano_midi (long)
    # MuseData (extra long)
    dataset = "JSB_Chorales"
    # traindata, testdata, validdata
    key = "traindata"
    index = 0 # which song in the set will be input

    # Supported architectures
    # LDS
    # TANH
    architecture = 'TANH'
    readout = 'linear'
    input_size = 88
    hidden_size = 300
    num_layers = 1
    output_size = 88

    sdpath = 'models/204/final_state_dict.pt'

    true_steps = 0 # how many time steps to copy from the original track
    input_steps = 100 # how many time steps will be the model prediction given the original track
    free_steps = 100 # how many time steps will be based on the output of the model alone
    history = 50 # how many past steps the model takes into account when synthesizing new music
    variance = 0.2 # variance of the noise meant to knock the system out of stable limit cycles

    # what to name the midi and waveform files
    song_name = "cyberbach"

    # whether or not we automatically use timidity to convert to wav
    convert2wav = True


# give all random number generators the same seed
def _seed_all(_seed) -> None:
    torch.manual_seed(_seed)
    np.random.seed(_seed)
    random.seed(_seed)


@ex.automain
def music_synthesis(dataset,
                    key,
                    index,
                    architecture,
                    readout,
                    input_size,
                    hidden_size,
                    num_layers,
                    output_size,
                    sdpath,
                    true_steps,
                    input_steps,
                    free_steps,
                    history,
                    variance,
                    song_name,
                    convert2wav,
                    _seed,
                    _log,
                    _run):

    _seed_all(_seed)

    # save artifacts to a temporary directory that gets erased when the experiment is over
    save_dir = 'tmp_' + str(_seed)
    os.system('mkdir ' + save_dir)
    save_dir += '/'

    # instructions for creating un-initialized model
    model_dict = {
                  'architecture': architecture,
                  'readout': readout,
                  'input_size': input_size,
                  'hidden_size': hidden_size,
                  'num_layers': num_layers,
                  'output_size': output_size,
                  'gradient_clipping': 1,
                 }

    # construct the model based on the saved state
    model = get_model(model_dict, {'init': "default"}, False)
    sd = torch.load(sdpath)
    model.load_state_dict(sd)

    # get the desired song
    track = loadmat("data/" + dataset)[key][0][index]

    # generate a new song
    new_track = make_music(model, track, true_steps, input_steps, free_steps, history, variance)

    # temporary name for this song
    track_name = save_dir + song_name + '.mid'

    # convert to midi and save in temporary directory
    to_midi(0, new_track, track_name)

    # make sacred remember where it is
    _run.add_artifact(track_name)

    if convert2wav:

        # create the wav file using timidity
        os.system("timidity -Ow " + track_name)

        # make sacred remember
        _run.add_artifact(save_dir + song_name + '.wav')

    # wait for a second then remove the temporary directory
    sleep(1)
    os.system("rm -r " + save_dir)