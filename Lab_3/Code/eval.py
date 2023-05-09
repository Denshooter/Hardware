import pandas as pd
import matplotlib.pyplot as plt

csv = pd.read_csv("benchmark.csv", names=["n", "seq", "simd"])

n = list(map(float, list(csv["n"])[1:]))
seq = list(map(float, list(csv["seq"])[1:]))
simd = list(map(float, list(csv["simd"])[1:]))

plt.figure()
plt.plot(n,seq, '.r', label="Sequential")
plt.plot(n,simd, '.b', label="SIMD")
plt.title("Matrix Multiplication")
plt.legend()
plt.ylabel("time [s]")
plt.xlabel("Matrix Dimension (N x N)")
plt.savefig("benchmark.png")
