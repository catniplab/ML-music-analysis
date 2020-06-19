"""
This script provides an example of how to call experiment.py
"""

from importlib import reload

import src.experiment
reload(src.experiment)
from src.experiment import ex  # importing experiment here is crucial (why?)

from sacred.observers import FileStorageObserver

import os

# custom configuration
config_updates = {}

# store in local directory for now
base_dir = os.getcwd()
ex.observers.append(FileStorageObserver(base_dir + '/results'))

ex.run(config_updates={**config_updates})
result = ex.current_run.result