"""
Test the random generation of square worlds.
"""
import unittest as test
import numpy as np
import matplotlib.pyplot as plt
import random_worlds as t

def testMountains(d=16, r=np.random.uniform, lo=13.0, hi=31.0):
    # lo and hi are noise factors for the center and edges respectively.
    lo = r(lo)
    hi = r(low=lo, high=hi)
    s = np.ceil(np.random.uniform(low=0, high=8*hi, size=(d+1,d+1)))
    t.mountains(s, d, nsc=hi, nse=lo)
    s = np.ceil(s)
    plt.imshow(s)
    plt.show()
    return s

def testClouds(d):
    s = np.ceil(np.random.uniform(0, 40, size=(d+1,d+1,d+1)))
    t.clouds(s, d)
    s = np.ceil(s)
    for i in range(min(32,d+1)):
        print("i=%d" % i)
        plt.imshow(s[:,:,i])
        plt.show()
    return s

if __name__ == "__main__":
    import os
    np.set_printoptions(linewidth=os.get_terminal_size().columns)
    #-np.set_printoptions(sign=' ')
    m = testMountains(256)
    c = testClouds(16)
