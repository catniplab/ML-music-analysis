"""
This script provides an example of how to call music_maker.py
"""

from importlib import reload

import math
import torch
import numpy as np

import src.music_maker
reload(src.music_maker)
from src.music_maker import ex  # importing experiment here is crucial (why?)

from sacred.observers import FileStorageObserver, RunObserver

import os

# don't record information in the file system, just investigate where the program fails
debug_mode = False


config_updates = {}


if __name__ == "__main__":

   base_dir = os.getcwd()

   if debug_mode:

      # run the experiment without an observer
      ex.run(config_updates={**config_updates})
      result = ex.current_run.result

   else:

      # store in local directory for now
      ex.observers.append(FileStorageObserver(base_dir + '/music'))

      # run the experiment after adding observer
      ex.run(config_updates={**config_updates})
      result = ex.current_run.result