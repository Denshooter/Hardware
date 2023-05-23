import matplotlib.pyplot as plot
from math import *

def read_data(filename):
    with open(filename, "r") as f:
        content = f.read()
        lines = content.split("\n")[:-1]
        points = [(float(line.split(",")[0]), float(line.split(",")[1])) for line in lines]
        return points

if __name__ == "__main__":
    # read the means
    means = read_data("means.csv")
    # read the data
    data = read_data("clusters.csv")

    # plot everything and save to png file
    datax, datay = list(zip(*data))
    meansx, meansy = list(zip(*means))
    plot.figure()
    plot.plot(datax, datay, 'b.')
    plot.plot(meansx, meansy, 'r.')
    plot.savefig("clustered.png")
