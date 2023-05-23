import matplotlib.pyplot as plot
import random
from math import *

class Cluster:
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var
    def sample(self):
        phi = random.uniform(0, pi * 2)
        d = random.uniform(0, self.var)
        x = d * sin(phi) + self.mean[0]
        y = d * cos(phi) + self.mean[1]
        return (x, y)
    def nsample(self, n):
        return [self.sample() for i in range(n)]
    def __repr__(self):
        return "<Cluster: mean = " + str(self.mean) + ", var = " + str(self.var) + " >"

def main():
    # create clusters
    c0 = Cluster((10, 5), 4)
    c1 = Cluster((30, 5), 9)
    c2 = Cluster((20, 15), 8)
    c3 = Cluster((35, 20), 5)

    # collect samples
    ps = []
    ps += c0.nsample(100)
    ps += c1.nsample(300)
    ps += c2.nsample(300)
    ps += c3.nsample(50)

    # shuffle the samples
    # the algorithm probably wouldn't work with sorted samples
    random.shuffle(ps)    

    # plot the data
    x, y = list(zip(*ps))
    plot.figure()
    plot.plot(x, y, '.')
    plot.savefig("data.png")

    # write samples to csv file
    with open("clusters.csv", "a") as f:
        for p in ps:
            f.write("%.2f, %.2f\n" % (p[0], p[1]))


random.seed(None)
if __name__ == "__main__":
    main()
