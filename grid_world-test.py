"""
Test the creation of grid-world Worlds.
"""
import numpy as np
import grid_world as gw


if __name__ == "__main__":
    import os
    np.set_printoptions(linewidth=os.get_terminal_size().columns)
    #-np.set_printoptions(sign=' ')

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

    print("\nAdjacent:\n")
    print(w.adjacent(np.array([7,7])))
