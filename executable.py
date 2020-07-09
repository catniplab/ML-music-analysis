"""
This script provides an example of how to call experiment.py
"""

from importlib import reload

import torch
import numpy as np

import src.experiment
reload(src.experiment)
from src.experiment import ex  # importing experiment here is crucial (why?)

from sacred.observers import FileStorageObserver, RunObserver

import os

# don't record information in the file system, just investigate where the program fails
debug_mode = False

# custom configuration
config_updates = {
                  'architecture': "REGRESSION_WIDE",
                  'optmzr': "SecondOrder",
                  'init': "default",
                  'parity': "rotate",

                  'num_epochs': 300,
                  'hps_epochs': 100,

                  'lag': 0,
                  'window': 7,

                  'decay': 0.93,
                  'lr': 0.01,

                  'ema_decay': 0.999,
                  'damping': 0.001,

                  'do_hpsearch': False,

                  'save_init_model': True,
                  'save_final_model': True
                 }

"""
# remove temporary directories when the experiment is over
class FileDeleter(RunObserver):

    def queued_event(self, ex_info, command, queue_time, config, meta_info, _id):
      pass

    def started_event(self, ex_info, command, host_info, start_time, config, meta_info, _id):
      pass

    def heartbeat_event(self, info, captured_out, beat_time, result):
      pass

    def completed_event(self, stop_time, result):
      print(result)

    def interrupted_event(self, interrupt_time, status):
      print(status)

    def failed_event(self, fail_time, fail_trace):
      print(self)

    def resource_event(self, filename):
      pass

    def artifact_event(self, name, filename):
      pass
"""

if __name__ == "__main__":

   base_dir = os.getcwd()
   #ex.observers.append(FileDeleter(base_dir))

   if debug_mode:

      # run the experiment without an observer
      ex.run(config_updates={**config_updates})
      result = ex.current_run.result

   else:

      # store in local directory for now
      ex.observers.append(FileStorageObserver(base_dir + '/results'))

      # run the experiment after adding observer
      ex.run(config_updates={**config_updates})
      result = ex.current_run.result