
import scipy.optimize as opt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

def sigmoid(x, a, h, c, k):  # upper limit at a+c
    return a / (1 + np.exp(-(x - h) / k)) + c

class SigmoidData(object):
    def __init__(self, data_x, data_y, concentration=None):
        self.concentration = concentration
        self.data_x = data_x
        self.data_y = data_y
        self.curve_fit()
        self.determine_important()

    def curve_fit(self):  # maximum a is the most important parameter. However, sigmoidal data might not be accurate
        optimized = opt.curve_fit(sigmoid, self.data_x, self.data_y,
                                  bounds=[[0, 0, -np.inf, -np.inf], [20, np.inf, np.inf, np.inf]])
        self.parameters = optimized[0]

    def plot(self):
        a, h, c, k = self.parameters
        key_xs = [self.inflection, self.bend_x_1]  # self.bend_x_0,
        z = np.linspace(np.min(self.data_x), np.max(self.data_x), 100)
        pred = sigmoid(z, a, h, c, k)

        plt.scatter(self.data_x, self.data_y, color="blue", alpha=0.1)
        plt.plot(z, pred, color="red")
        plt.scatter(key_xs, sigmoid(key_xs, a, h, c, k), color="green")
        plt.plot(self.data_x, [self.curve_max for x in self.data_x], color="purple", alpha=0.5)

        print("Parameters: " +str(self.parameters))
        print("R^2: " + str(self.r2))
        print("Curve Fitted Maximum: " + str(self.curve_max))
        print("Inflection Point: " + str(self.inflection))
        print("Spikiness: " + str(self.spikiness))
        print("K: " + str(k))
        plt.show()

    def determine_important(self):
        a, h, c, k = self.parameters
        self.curve_max = a + c  # Y value
        self.inflection = k * np.log(np.exp(2 * h / k)) - h
        self.bend_x_0 = k * np.log(-(2 * np.sqrt(6) - 5) * np.exp(h / k))
        self.bend_x_1 = k * np.log((2 * np.sqrt(6) + 5) * np.exp(h / k))
        self.spikiness = self.bend_x_1 - self.bend_x_0
        self.r2 = r2_score(self.data_y, sigmoid(self.data_x, a, h, c, k))
