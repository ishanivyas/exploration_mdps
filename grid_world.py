"""
Creating a 2D and 3D grid environment for agent to move around.
Created By Ishani Vyas
"""
import numpy as np
import random_worlds as rw
from math import log2,ceil

class World:
    def __init__(self, r=np.random):
        self.r = r

    def distance(self, s0, s1):
        """Return the (scalar) distance between the two states."""
        pass

    def contains(self, s):
        """Return True if state is within the bounds of the World."""
        pass

    def adjacent(self, s):
        """Return the actions avalaible from state s and the states they reach."""
        pass

    def actions_allowed(self, s):
        """Return the available actions from `s`."""
        return self.adjacent(s)[0]

    def enumerateStates(self):
        """List all of the possible states in the World."""
        pass

    def transition(self, s, a):
        """
            Returns the next state after taking action `a` from state `s`.
            NOTE: for some Worlds this may not be deterministic.
        """
        return tuple(np.add(s, a))


class Grid(World):
    def __init__(self, r=np.random):
        super(Grid, self).__init__(r)

    def contains(self, s):
        """Return True if state is within the bounds of the World."""
        return np.amin(s) >= 0 and np.amin(self.bounds - s) > 0

    def distance(self, s0, s1):
        """Return the Manhattan distance between the two states but allow diagonal moves."""
        d = np.abs(s0 - s1)  # Manhattan distance...
        return np.amax(d)    # ...allowing diagonals.

    def randomState(self):
        """Returns the coordinates of a random state."""
        return tuple(np.floor(self.bounds
                              * self.r.uniform(size=self.bounds.shape)))


class Grid2D(Grid):
    """
    2D grid world that allows diagonal moves.
    """
    def __init__(self, widthOrArray, height=None, maxRange=None, r=np.random):
        """
        Examples:
          w = Grid2D(3,4, 8)  # A 3x4 grid
          w = Grid2D(np.array([[1,2],
                               [2,3]]))
        """
        super(Grid2D, self).__init__(r)
        self.terminalState  = 'TERMINAL_STATE'
        self.state_dim      = 2  # The number of coordinates needed to unambiguously define an agent's state in the world.
        self.max_action_dim = pow(3, self.state_dim) - 1

        if isinstance(widthOrArray, np.ndarray):
            self.data   = widthOrArray
            self._initBounds(self.data.shape)
        else:
            self._initBounds(np.array([widthOrArray, height]))
            ur           = self.r.uniform
            rCenterScale = ur(13.0)  # The factor to reduce noise by in the center.
            rEdgeScale   = ur(low=rCenterScale, high=ur(31.0))  # The factor to reduce noise by along edges/faces.
            d            = 2**ceil(log2(np.amax(self.bounds)))
            space        = ur(maxRange, size=(d+1,d+1))
            rw.mountains(space, d, r=ur, nsc=rCenterScale, nse=rEdgeScale)
            self.data = np.round(space[0:widthOrArray,0:height])

    def _initBounds(self, bounds):
        self.bounds = bounds
        self.width  = bounds[0]
        self.height = bounds[1]

    def __getitem__(self, i):
        return self.data[i]

    def __setitem__(self, key, item):
        self.data[key] = item

    def __eq__(self, other):
        if other == None: return False
        return self.data == other.data

    def __hash__(self):
        return hash(self.data)

    def copy(self):
        g = Grid2D(self.width, self.height)
        g.data = [x[:] for x in self.data]
        return g

    def deepCopy(self):
        return self.copy()

    def shallowCopy(self):
        g = Grid2D(self.width, self.height)
        g.data = self.data
        return g

    def adjacent(self, s):
        """Return the actions avalaible from state s and the states they reach."""
        adj = []
        act = []
        d = [-1,0,1]
        for dy in d:
            for dx in d:
                action = np.array([dx, dy])
                s_prime = s + action
                if self.distance(s, s_prime) == 0 or not self.contains(s_prime):
                    continue
                act.append(tuple(action))
                adj.append(s_prime)
        return tuple(act),tuple(adj)

    def _getLegacyText(self):
        t = [[self.data[x][y] for x in range(self.width)] for y in range(self.height)]
        t.reverse()
        return t

    def __str__(self):
        return str(self._getLegacyText())

    def enumerateStates(self):
        """Returns a list of all possible state coordinates [(x, y), ...]"""
        states = []
        for i in range(self.width):
            for j in range(self.height):
                states.append((i, j))
        return states


class OrthoGrid2D(Grid2D):
    """2D grid that only allows othogonal moves."""
    def __init__(self, widthOrArray, height=None, maxRange=None, r=np.random):
        super(OrthoGrid2D, self).__init__(widthOrArray, height, maxRange, r)
        self.max_action_dim = 2*self.state_dim

    def distance(self, s0, s1):
        return np.sum(np.abs(s0 - s1))

    def adjacent(self, s):
        adj = []
        act = []
        for dx in [-1,1]:
            action = np.array([dx, 0])
            s_prime = s + action
            if not self.contains(s_prime): continue
            act.append(tuple(action))
            adj.append(s_prime)

        for dy in [-1,1]:
            action = np.array([0, dy])
            s_prime = s + action
            if not self.contains(s_prime): continue
            act.append(tuple(action))
            adj.append(s_prime)

        return tuple(act),tuple(adj)


class Grid3D(Grid):
    """
        3D grid world that allows diagonal moves.
    """
    def __init__(self, widthOrArray, heightOrDeepCopy=None, depth=None, maxRange=None, r=np.random):
        """
        Examples:
          w = Grid3D(3,4,5, 8)
          w = Grid3D(np.array([[[1,2], [2,2]],
                               [[1,2], [3,4]]]))
        """
        super(Grid3D, self).__init__(r)
        self.terminalState  = 'TERMINAL_STATE'
        self.state_dim      = 3   # The number of coordinates needed to unambiguously define an agent's state in the world.
        self.max_action_dim = pow(3, self.state_dim) - 1

        if isinstance(widthOrArray, np.ndarray):
            if heightOrDeepCopy: # widthOrArray is array, so this is really deepCopy
                self.data = np.array(widthOrArray)
            else:
                self.data = widthOrArray
            self._initBounds(self.data.shape)
        else:
            self._initBounds(np.array([widthOrArray, heightOrDeepCopy, depth]))
            ur           = self.r.uniform
            rCenterScale = ur(13.0)  # The factor to reduce noise by in the center.
            rEdgeScale   = ur(low=rCenterScale, high=ur(31.0))  # The factor to reduce noise by along edges/faces.
            d            = 2**ceil(log2(np.amax(self.bounds)))
            space        = ur(maxRange, size=(d+1, d+1, d+1))
            rw.clouds(space, d, r=ur, nsc=rCenterScale, nse=rEdgeScale)
            self.data = np.round(space[0:widthOrArray, 0:heightOrDeepCopy, 0:depth])

    def _initBounds(self, bounds):
        self.bounds = bounds
        self.width  = bounds[0]
        self.height = bounds[1]
        self.depth  = bounds[2]

    def copy(self):
        """Copy the data of a 3D grid into a new instance of Grid3D."""
        return Grid3D(self.data, True)

    def deepCopy(self):
        return self.copy()

    def shallowCopy(self):
        return Grid3D(self.data, False)

    def adjacent(self, s):
        """Return the actions available from state s and the states they reach."""
        act = []
        adj = []
        d = [-1,0,1]
        for dz in d:
            for dy in d:
                for dx in d:
                    if dx == dy == dz == 0: continue
                    action = np.array([dx,dy,dz])
                    s_prime = s + action
                    if np.amin(s_prime) < 0: continue
                    if self.distance(s, s_prime) == 0 or not self.contains(s_prime):
                        continue
                    act.append(action)
                    adj.append(s_prime)
        return tuple(act),tuple(adj)

    def enumerateStates(self):
        pass

    def __str__(self):
        return str(self.data)


class OrthoGrid3D(Grid3D):
    """3D grid that only allows othogonal moves."""
    def __init__(self, widthOrArray, height=None, maxRange=None, r=np.random):
        super(OrthoGrid3D, self).__init__(widthOrArray, height, maxRange, r)
        self.max_action_dim = 2*self.state_dim

    def distance(self, s0, s1):
        return np.sum(np.abs(s0 - s1))

    def adjacent(self, s):
        adj = []
        act = []
        for dx in [-1,1]:
            action = np.array([dx, 0, 0])
            s_prime = s + action
            if not self.contains(s_prime): continue
            act.append(tuple(action))
            adj.append(s_prime)

        for dy in [-1,1]:
            action = np.array([0, dy, 0])
            s_prime = s + action
            if not self.contains(s_prime): continue
            act.append(tuple(action))
            adj.append(s_prime)

        for dz in [-1,1]:
            action = np.array([0, 0, dz])
            s_prime = s + action
            if not self.contains(s_prime): continue
            act.append(tuple(action))
            adj.append(s_prime)

        return tuple(act),tuple(adj)
