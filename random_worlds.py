"""
Generate random worlds that are squares/cubes of size 2^n.
"""
import os
import numpy as np

np.set_printoptions(linewidth=os.get_terminal_size().columns)
#-np.set_printoptions(sign=' ')

def mountains(s, d, x, y, r=np.random.uniform, nsc=3.0, nse=27.0):
    """Apply the diamond-square algorithm to produce random mountains."""
    #   See: https://en.wikipedia.org/wiki/Diamond-square_algorithm
    h = d//2  # Calc the half-dimension using integer division.
    s[x+h,y+h] = c = (s[x,y] + s[x+d,y] + s[x,y+d] + s[x+d,y+d])/4 + r(d/nsc)  # Center-middle
    # Calculate the edge values.
    s[x+h,y+0] = (s[x+0,y+0] + s[x+d,y+0])/2 + r(d/nse)  # Top-center
    s[x+0,y+h] = (s[x+0,y+0] + s[x+0,y+d])/2 + r(d/nse)  # Left-middle
    s[x+h,y+d] = (s[x+d,y+d] + s[x+0,y+d])/2 + r(d/nse)  # Bottom-center
    s[x+d,y+h] = (s[x+d,y+d] + s[x+d,y+0])/2 + r(d/nse)  # Bottom-middle
    # Recursively calculate the values for the 4 sub-squares.
    if h >= 2:
        mountains(s, h, x+0, y+0, r, nsc, nse)
        mountains(s, h, x+0, y+h, r, nsc, nse)
        mountains(s, h, x+h, y+0, r, nsc, nse)
        mountains(s, h, x+h, y+h, r, nsc, nse)

def clouds(s, d, x, y, z, r=np.random.uniform, nsc=3.0, nsf=27.0):
    """Apply the diamond-square algorithm to produce random clouds."""
    #   See: https://en.wikipedia.org/wiki/Diamond-square_algorithm
    h = d//2  # Calc the half-dimension using integer division.
    s[x+h,y+h,z+h] = (s[x,y,z]   + s[x+d,y,z]   + s[x,y+d,z]   + s[x+d,y+d,z] + s[x,y,z+d] + s[x+d,y,z+d] + s[x,y+d,z+d] + s[x+d,y+d,z+d])/8 + r(d/nsc)  # Center-middle
    # Calculate the 6 face-centered values.
    s[x+h,y+h,z+0] = (s[x+0,y+0,z+0] + s[x+0,y+d,z+0] + s[x+d,y+d,z+0] + s[x+d,y+0,z+0])/4 + r(d/nsf)  # XY plane, Z=z+0
    s[x+h,y+0,z+h] = (s[x+0,y+0,z+0] + s[x+d,y+0,z+0] + s[x+d,y+0,z+d] + s[x+0,y+0,z+d])/4 + r(d/nsf)  # XZ plane, Y=y+0
    s[x+0,y+h,z+h] = (s[x+0,y+0,z+0] + s[x+0,y+d,z+0] + s[x+0,y+d,z+d] + s[x+0,y+0,z+d])/4 + r(d/nsf)  # YZ plane, X=x+0
    s[x+h,y+h,z+d] = (s[x+d,y+d,z+d] + s[x+d,y+0,z+d] + s[x+0,y+0,z+d] + s[x+0,y+d,z+d])/4 + r(d/nsf)  # XY plane, Z=z+d
    s[x+h,y+d,z+h] = (s[x+d,y+d,z+d] + s[x+0,y+d,z+d] + s[x+0,y+d,z+0] + s[x+d,y+d,z+0])/4 + r(d/nsf)  # XZ plane, Y=y+d
    s[x+d,y+h,z+h] = (s[x+d,y+d,z+d] + s[x+d,y+0,z+d] + s[x+d,y+0,z+0] + s[x+d,y+d,z+0])/4 + r(d/nsf)  # YZ plane, X=x+d
    # Calculate the 12 edge-centered values.
    s[x+0,y+0,z+h] = (s[x+0,y+0,z+0] + s[x+0,y+0,z+d])/2 + r(d/nsf)
    s[x+0,y+h,z+0] = (s[x+0,y+0,z+0] + s[x+0,y+d,z+0])/2 + r(d/nsf)
    s[x+h,y+0,z+0] = (s[x+0,y+0,z+0] + s[x+d,y+0,z+0])/2 + r(d/nsf)
    s[x+d,y+d,z+h] = (s[x+d,y+d,z+d] + s[x+d,y+d,z+0])/2 + r(d/nsf)
    s[x+d,y+h,z+d] = (s[x+d,y+d,z+d] + s[x+d,y+0,z+d])/2 + r(d/nsf)
    s[x+h,y+d,z+d] = (s[x+d,y+d,z+d] + s[x+0,y+d,z+d])/2 + r(d/nsf)
    s[x+0,y+d,z+h] = (s[x+0,y+d,z+d] + s[x+0,y+d,z+0])/2 + r(d/nsf)
    s[x+d,y+0,z+h] = (s[x+d,y+0,z+d] + s[x+d,y+0,z+0])/2 + r(d/nsf)
    s[x+0,y+h,z+d] = (s[x+0,y+d,z+d] + s[x+0,y+0,z+d])/2 + r(d/nsf)
    s[x+d,y+h,z+0] = (s[x+d,y+d,z+0] + s[x+d,y+0,z+0])/2 + r(d/nsf)
    s[x+h,y+0,z+d] = (s[x+d,y+0,z+d] + s[x+0,y+0,z+d])/2 + r(d/nsf)
    s[x+h,y+d,z+0] = (s[x+d,y+d,z+0] + s[x+0,y+d,z+0])/2 + r(d/nsf)
    # Recursively calculate the values for the 8 sub-cubes.
    if h >= 2:
        clouds(s, h, x+0, y+0, z+0, r, nsc, nsf)
        clouds(s, h, x+0, y+0, z+h, r, nsc, nsf)
        clouds(s, h, x+0, y+h, z+0, r, nsc, nsf)
        clouds(s, h, x+0, y+h, z+h, r, nsc, nsf)
        clouds(s, h, x+h, y+0, z+0, r, nsc, nsf)
        clouds(s, h, x+h, y+0, z+h, r, nsc, nsf)
        clouds(s, h, x+h, y+h, z+0, r, nsc, nsf)
        clouds(s, h, x+h, y+h, z+h, r, nsc, nsf)

