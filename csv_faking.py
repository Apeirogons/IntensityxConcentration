import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from SigmoidData import SigmoidData, sigmoid

def curve_fit_from_csv(csv):
    readout = pd.read_csv(csv)
    xs = readout["Time"]
    ys = readout["Intensity"]
    reversed_name = csv[::-1]
    last_slash = reversed_name.find("/")
    last_slash_loc = len(csv) - last_slash
    name = float(csv[last_slash_loc:csv.find("_")])
    return SigmoidData(xs, ys, name)


def create_fake_csv(csv, xs, ys):
    readout = pd.DataFrame()
    readout["Time"] = xs
    readout["Intensity"] = ys
    readout.to_csv(csv)


def create_many_fake_csvs(n_experiment_runs, folder, a, h, c, k, noise=0.2, plotting = True):
    xs = np.linspace(0, 20, 100)  # ex. 100 data points between 0 and 20 minutes
    experiment_i_conc = np.linspace(0, 10, n_experiment_runs)  # initial concentration of a bunch of experiments
    a_distribution = sigmoid(experiment_i_conc, a, h, c, k) + np.random.normal(scale=noise,
                                                                               size=experiment_i_conc.shape[
                                                                                   0])  # a is distributed sigmoidally with a given sigmoid curve
    if not os.path.exists(folder):
        os.mkdir(folder)
    if plotting:
        plt.scatter(experiment_i_conc, a_distribution)
        plt.show()
    for n in range(n_experiment_runs):
        ys = sigmoid(xs, a_distribution[n], 5, 0, 1) + np.random.normal(scale=0.1, size=xs.shape[0])
        create_fake_csv(folder + "/" + str(experiment_i_conc[n]) + "_mM.csv", xs, ys)