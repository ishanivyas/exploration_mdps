"""
Test the random generation of square worlds.
"""
import unittest as test
import numpy as np
import matplotlib.pyplot as plt
import random_worlds as t

class RandomWorldsTest(test.TestCase):
    def testMountainsWithoutNoiseIsAverage(self):
        # 3x3 (d=2)
        s = np.array([[1,0,3],
                      [0,0,0],
                      [3,0,5]])
        t.mountains(s, 2, r=lambda _: 0)
        np.testing.assert_array_equal(
            s,
            np.array([[1,2,3],
                      [2,3,4],
                      [3,4,5]])
        )

        # 5x5 (d=4)
        s = np.array([[1,0,0,0,5],
                      [0,0,0,0,0],
                      [0,0,0,0,0],
                      [0,0,0,0,0],
                      [5,0,0,0,9]])
        t.mountains(s, 4, r=lambda _: 0)
        np.testing.assert_array_equal(
            s,
            np.array([[1,2,3,4,5],
                      [2,3,4,5,6],
                      [3,4,5,6,7],
                      [4,5,6,7,8],
                      [5,6,7,8,9]])
        )

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

    test.main()
