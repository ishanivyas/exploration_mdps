import numpy as np
import grid_world as gw

if __name__ == "__main__":
    gw = gw.Grid2D(32, 64, 8)
    print(gw.data)

    d = gw.distance(np.array([1,1]), np.array([3,6]))
    print(d)
