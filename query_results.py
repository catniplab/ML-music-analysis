import os
import json

# find the directories with these configurations
config_dict = {
              'training': {
                          'dataset': "JSB_Chorales",
                          'num_epochs': 150,
                          'batch_size': 128,
                          'lr': 0.001,
                          'optimizer': "SGD"
                          }
              }

# success argument checks if there are NaNs in the loss records
def find_results(configs, success=False):

    good_dirs = []

    dirs = os.listdir('results')

    for dir in dirs:

        if dir != "_sources":

            config_file = open('results/' + dir + '/config.json')
            config_contents = config_file.read()
            config_file.close()

            file_configs = json.loads(config_contents)

            agree = True

            for key, value in configs.items():
                if key in file_configs.keys():

                    val = file_configs[key]
                    if type(value) == dict and type(val) == dict:
                        for k, v in value.items():
                            if k in val.keys():
                                if val[k] == v:
                                    continue
                                else:
                                    agree = False
                                    break

                    elif file_configs[key] == value:
                        continue
                    else:
                        agree = False
                        break
                else:
                    agree = False
                    break

            if success:

                metric_file = open('results/' + dir + '/metrics.json')
                metric_contents = metric_file.read()
                metric_file.close()

                metrics = json.loads(metric_contents)
                trainLoss = metrics['trainLoss']['values']
                testLoss = metrics['testLoss']['values']
                validLoss = metrics['validLoss']['values']

                nan = float("NaN")

                if nan in trainLoss or nan in testLoss or nan in validLoss:
                    agree = False

            if agree:
                good_dirs.append(dir)

    return good_dirs

if __name__ == "__main__":
    print(find_results(config_dict, success=True))