"""
Test the learning of agents.
"""
import unittest as test
import numpy as np
import agent as a
import grid_world as gw

class AgentsTest(test.TestCase):
    def testRandomAgent(self):
        rs  = np.random.RandomState(seed=1)
        env = gw.Grid2D(np.array([[1,2,3],
                                  [2,3,4],
                                  [3,4,5]]), r=rs)
        eg  = a.EpsilonGreedyAgent(env, r=rs)

if __name__ == "__main__":
    test.main()
