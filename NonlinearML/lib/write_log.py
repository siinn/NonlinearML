import os
import datetime
import pandas as pd

import NonlinearML.lib.io as io


def write_log(list_features, input_path, log_path, model, grid_train_results, grid_test_results,
              test_begin, test_end):
    """ Write configurations such as train, test time period, features, etc.
    Args:
        list_feature: list of features
        input_path: input dataframe
        log_path: path to save logs
        model: Keras_Model class object
        grid_train_results: model evaluation results obtained from train dataset
        grid_test_results: model evaluation results obtained from test dataset
        train, test dates: start and end dates of train, test dataset
    Return:
        None
    """
    # Create folder to save log
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    # Open file to write log
    io.message("Writing results to %s" % datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    with open(log_path+"tf_%s" % datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"), "a") as f:
        f.write("==================================\n")
        f.write(" Keras Model\n")
        f.write("==================================\n")
        f.write(" > datetime = %s\n" % datetime.datetime.now())
        f.write(" > input feature dataframe: %s\n" % input_path)

        f.write("\n==================================\n")
        f.write(" Features\n")
        f.write("==================================\n")
        for feature in list_features:
            f.write("\t- %s\n" % feature)
        f.write(" > Total number of features: %s\n" % len(list_features))

        f.write("\n==================================\n")
        f.write(" Train and test split\n")
        f.write("==================================\n")
        f.write(" > test begin: %s\n" % test_begin)
        f.write(" > test end: %s\n" % test_end)

        f.write("\n==================================\n")
        f.write(" Model summary\n")
        f.write("==================================\n")
        f.write(" > summary: \n")
        model.model.summary(line_length=100, print_fn = lambda x: f.write(' > ' + x + '\n'))
        f.write(" > details: \n%s\n" % model.model.to_yaml())
        f.write(" > json summary: %s\n" % model.model.to_json())
        f.write("\n==================================\n")
        f.write(" Evaluation summary\n")
        f.write("==================================\n")
        with pd.option_context('max_colwidth', 100, 'expand_frame_repr', False, 'display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            f.write("\nTrain results summary\n")
            f.write(str(grid_train_results))
            f.write("\nTest results summary\n")
            f.write(str(grid_test_results))
    return
