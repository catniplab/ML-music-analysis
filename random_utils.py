import os
import json
import numpy as np

def get_losses(dirs):

    result = []

    for name in dirs:

        handle = open('results/' + str(name) + '/metrics.json')
        my_dict = json.loads(handle.read())
        handle.close()

        result.append(my_dict['trainLoss']['values'][-1])

    return result
