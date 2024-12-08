import matplotlib.pyplot as plt
import numpy as np


def gaussian(x, mean, std):
    return (1/(std*np.sqrt(np.pi * 2))) * np.exp((-1/2)*((x-mean)/std)**2)


x1 = [-1, 0, -1, 0, -1, 0, -1, -1, 0, 0]
x2 = [-1,-1,-1,0,-1,0,0,-1,-1,0]


mu1 = np.mean(x1)
mu2 = np.mean(x2)

std1 = np.std(x1)
std2 = np.std(x2)

x = np.linspace(-2, 2, 1000)

plt.plot(x,gaussian(x,mu1,std1))

plt.plot(x,gaussian(x,mu2,std2))

plt.title("Distribution of bias for articles and summaries")
plt.xlabel("Political bias")
plt.ylabel("Probability")

plt.savefig("Show.png")