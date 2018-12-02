"""
Test the creation of grid-world Worlds.
"""
import unittest as test
import numpy as np
import grid_world as gw

class GridTest(test.TestCase):
    def testContains(self):
        w2d = gw.Grid2D(np.zeros((8,8)))
        self.assertTrue( w2d.contains(np.array([0,0])))
        self.assertTrue( w2d.contains(np.array([7,7])))
        self.assertFalse(w2d.contains(np.array([0,-1])))
        self.assertFalse(w2d.contains(np.array([7,8])))
        self.assertFalse(w2d.contains(np.array([8,0])))

        w3d = gw.Grid3D(np.zeros((8,8,4)))
        self.assertTrue( w3d.contains(np.array([0,0,0])))
        self.assertTrue( w3d.contains(np.array([7,7,3])))
        self.assertFalse(w3d.contains(np.array([0,0,-1])))
        self.assertFalse(w3d.contains(np.array([7,8,3])))
        self.assertFalse(w3d.contains(np.array([8,0,4])))

    def testDistance(self):
        w2d = gw.Grid2D(np.zeros((8,8)))
        self.assertEqual(w2d.distance(np.array([0,0]),
                                      np.array([3,6])),
                         6)

        w3d = gw.Grid3D(np.zeros((8,8,4)))
        self.assertEqual(w2d.distance(np.array([0,0,0]),
                                      np.array([3,6,3])),
                         6)

    def testAdjacent2D(self):
        w2d = gw.Grid2D(np.zeros((3,3)))
        actions,adj = w2d.adjacent(np.array([1,1]))
        actions = [list(a) for a in actions]
        self.assertCountEqual(actions, [
            #-[ 0, 0],
            [ 0, 1],
            [ 0,-1],
            [ 1, 0],
            [ 1, 1],
            [ 1,-1],
            [-1, 0],
            [-1, 1],
            [-1,-1],
        ])

        adj = [list(a) for a in adj]
        self.assertCountEqual(adj, [
            [ 0, 0],
            [ 0, 1],
            [ 0, 2],
            [ 1, 0],
            #-[ 1, 1],
            [ 1, 2],
            [ 2, 0],
            [ 2, 1],
            [ 2, 2],
        ])

    def testAdjacent3D(self):
        w3d = gw.Grid3D(np.zeros((3,3,3)))
        actions,adj = w3d.adjacent(np.array([1,1,1]))
        actions = [list(a) for a in actions]
        self.assertCountEqual(actions, [
            #-[ 0, 0, 0],
            [ 0, 0, 1],
            [ 0, 0,-1],
            [ 0, 1, 0],
            [ 0, 1, 1],
            [ 0, 1,-1],
            [ 0,-1, 0],
            [ 0,-1, 1],
            [ 0,-1,-1],

            [ 1, 0, 0],
            [ 1, 0, 1],
            [ 1, 0,-1],
            [ 1, 1, 0],
            [ 1, 1, 1],
            [ 1, 1,-1],
            [ 1,-1, 0],
            [ 1,-1, 1],
            [ 1,-1,-1],

            [-1, 0, 0],
            [-1, 0, 1],
            [-1, 0,-1],
            [-1, 1, 0],
            [-1, 1, 1],
            [-1, 1,-1],
            [-1,-1, 0],
            [-1,-1, 1],
            [-1,-1,-1],
        ])

        adj = [list(a) for a in adj]
        self.assertCountEqual(adj, [
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 2],
            [0, 1, 0],
            [0, 1, 1],
            [0, 1, 2],
            [0, 2, 0],
            [0, 2, 1],
            [0, 2, 2],

            [1, 0, 0],
            [1, 0, 1],
            [1, 0, 2],
            [1, 1, 0],
            #-[1, 1, 1],
            [1, 1, 2],
            [1, 2, 0],
            [1, 2, 1],
            [1, 2, 2],

            [2, 0, 0],
            [2, 0, 1],
            [2, 0, 2],
            [2, 1, 0],
            [2, 1, 1],
            [2, 1, 2],
            [2, 2, 0],
            [2, 2, 1],
            [2, 2, 2],
        ])

if __name__ == "__main__":
    import os
    np.set_printoptions(linewidth=os.get_terminal_size().columns)
    #-np.set_printoptions(sign=' ')
    np.random.seed(1)

    print("SmallWorld:")
    sw = gw.Grid2D(np.arange(64).reshape((8,8)))
    sw.data[:,0] = np.arange(sw.data.shape[0])
    sw.data[0,:] = np.arange(sw.data.shape[1])
    print(sw.data)

    print("\nWorld:")
    w = gw.Grid2D(16, 16, 8)
    w.data[:,0] = np.arange(w.data.shape[0])
    w.data[0,:] = np.arange(w.data.shape[1])
    print(w.data)

    print("\nDistance between <1,1> and <3,6>:")
    d = w.distance(np.array([1,1]), np.array([3,6]))
    print(d)

    print("\nAdjacent to <7,7>:")
    act,adj = w.adjacent(np.array([7,7]))
    print("  Actions: %s" % str(act))
    print("  Adjacent States: %s" % str(adj))

    test.main()
