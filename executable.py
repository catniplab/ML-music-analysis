"""
This script provides an example of how to call experiment.py
"""

from importlib import reload

import src.experiment
reload(src.experiment)
from src.experiment import ex  # importing experiment here is crucial (why?)

from sacred.observers import FileStorageObserver, RunObserver

import os

# don't observe, just investigate where the program fails
debug_mode = False

# custom configuration
config_updates = {
                  'training': {
                              'dataset': "JSB_Chorales",
                              'num_epochs': 150,
                              'batch_size': 128,
                              'lr': 0.001,
                              'decay': 0.96,
                              'optimizer': "SGD"
                              },
                  'initializer': {
                                 'init': 'default',
                                 'scale': 1.0,
                                 'min_angle': 0.0,
                                 'max_angle': 2.0
                                 },
                   'model_dict': {
                                  'architecture': 'LINEAR',
                                  'gradient_clipping': 1,
                                  'jit': False,
                                  'input_size': 88,
                                  'hidden_size': 300,
                                  'num_layers': 1,
                                  'output_size': 88
                                 },
                   'detect_anomaly': True

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