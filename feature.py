"""
Tile Coding Software version 3.0beta
by Rich Sutton
based on a program created by Steph Schaeffer and others
External documentation and recommendations on the use of this code is available in the
reinforcement learning textbook by Sutton and Barto, and on the web.
These need to be understood before this code is.

This software is for Python 3 or more.

This is an implementation of grid-style tile codings, based originally on
the UNH CMAC code (see http://www.ece.unh.edu/robots/cmac.htm), but by now highly changed.
Here we provide a function, "tiles", that maps floating and integer
variables to a list of tiles, and a second function "tiles-wrap" that does the same while
wrapping some floats to provided widths (the lower wrap value is always 0).

The float variables will be gridded at unit intervals, so generalization
will be by approximately 1 in each direction, and any scaling will have
to be done externally before calling tiles.

Num-tilings should be a power of 2, e.g., 16. To make the offsetting work properly, it should
also be greater than or equal to four times the number of floats.

The first argument is either an index hash table of a given size (created by (make-iht size)),
an integer "size" (range of the indices from 0), or nil (for testing, indicating that the tile
coordinates are to be returned without being converted to indices).
"""

from math import floor, log
from itertools import zip_longest
import numpy as np


#############################################################################################
#                                          1. Tile Coding Utils                             #
#############################################################################################
# Credit : http://www.incompleteideas.net/tiles/tiles3.html

basehash = hash

class IHT:
    """Structure to handle collisions."""

    def __init__(self, sizeval):
        self.size = sizeval
        self.overfullCount = 0
        self.dictionary = {}

    def __str__(self):
        """Prepares a string for printing whenever this object is printed."""
        return "Collision table:" + \
               " size:" + str(self.size) + \
               " overfullCount:" + str(self.overfullCount) + \
               " dictionary:" + str(len(self.dictionary)) + " items"

    def count(self):
        return len(self.dictionary)

    def fullp(self):
        return len(self.dictionary) >= self.size

    def getindex(self, obj, readonly=False):
        d = self.dictionary
        if obj in d:
            return d[obj]
        elif readonly:
            return None
        size = self.size
        count = self.count()
        if count >= size:
            if self.overfullCount == 0: print('IHT full, starting to allow collisions')
            assert self.overfullCount != 0
            self.overfullCount += 1
            return basehash(obj) % self.size
        else:
            d[obj] = count
            return count

class IndexHashTable:

    def __init__(self, iht_size, num_tilings, tiling_size, obs_bounds):
        # Index Hash Table size
        self._iht = IHT(iht_size)
        # Number of tilings
        self._num_tilings = num_tilings
        # Tiling size
        self._tiling_size = tiling_size
        # Observation boundaries
        # (format : [[min_1, max_1], ..., [min_i, max_i], ... ] for i in state's components)
        self._obs_bounds = obs_bounds

    def get_tiles(self, state, action):
        """Get the encoded state_action using Sutton's grid tiling software."""
        # List of floats numbers to be tiled
        floats = [s * self._tiling_size / (obs_max - obs_min)
                  for (s, (obs_min, obs_max)) in zip(state, self._obs_bounds)]
        # print("floats", floats)
        return tiles(self._iht, self._num_tilings, floats, [action])


class CartPoleTileCoding:
    """
    Tile coding the cartpole environment
    """
    def __init__(self, num_tilings=8, tiling_size=1, iht_size=2**10):
        # Observation boundarie
        # (format : [[min_1, max_1], ..., [min_i, max_i], ... ] for i in state's components.
        #  state = (x, x_dot, theta, theta_dot)
        #  "Fake" bounds have been set for velocity components to ease tiling.)
        obs_bounds = [[-4.8, 4.8],
                      [-3., 3.],
                      [-0.25, 0.25],
                      [-3., 3.]]
        # Tiling parameters
        self._iht_args = {'iht_size': iht_size,
                          'num_tilings': num_tilings,
                          'tiling_size': tiling_size,
                          'obs_bounds': obs_bounds}

    def get_tile_coding_args(self):
        return self._iht_args


class TabularTileCoding:
    """
    Tile coding arguments of Tabular environment
    """
    def __init__(self, iht_size, num_tilings, tiling_size):
        # Observation boundaries
        # (format : [[min_1, max_1], ..., [min_i, max_i], ... ] for i in state's components.
        #  state = (x, x_dot, theta, theta_dot)
        #  "Fake" bounds have been set for velocity components to ease tiling.)
        # obs_bounds = [[0,5],[0,5]] # bounds for value of state
        obs_bounds = [[0, 21]]  # bounds for value of state
        # Tiling parameters
        self._iht_args = {'iht_size': iht_size, # size of iht map
                          'num_tilings': num_tilings, # number of such grids, returns same number of non-zero 1s
                          'tiling_size': tiling_size, # constructs a [tiling_size X tiling_size] grid
                          'obs_bounds': obs_bounds}

    def get_tile_coding_args(self):
        return self._iht_args

    def get_state_representation(self, state):
        """
        Maps [0-24] state to 5X5 grid.
        """
        return [state]


class Features:
    """
    Feature class implements the one-hot encoding features
    """
    def __init__(self, num_actions):
        self.A = num_actions

    def get_one_hot_encoding(self, s, a):
        # expanded features in row format
        return int(s * self.A + a)

    def get_feature(self, s, a):
        return [self.get_one_hot_encoding(s, a)]


class TileCodingFeatures(Features):
    """
    Tile Coding class which returns the feature given (s,a) pair for tabular MDP
    """

    def __init__(self, num_actions, iht_args):
        self.iht = IndexHashTable(**iht_args)
        self.num_features = iht_args['iht_size']
        self.num_tiles = iht_args['num_tilings']
        super().__init__(num_actions)

    def get_feature(self, s, a):
        # return the tiles form tile coding function given (state,action) pair
        return self.iht.get_tiles(s, a)

    def get_feature_size(self):
        return self.num_features

    def get_num_tiles(self):
        return self.num_tiles
    
    def one_hot(self, d, feature):
        tmp_feature = np.zeros(d)
        tmp_feature[feature] = 1
        return tmp_feature



def hashcoords(coordinates, m, readonly=False):
    if type(m) == IHT: return m.getindex(tuple(coordinates), readonly)
    if type(m) == int: return basehash(tuple(coordinates)) % m
    if m == None: return coordinates


def tiles(ihtORsize, numtilings, floats, ints=[], readonly=False):
    """Returns num-tilings tile indices corresponding to the floats and ints"""
    qfloats = [floor(f * numtilings) for f in floats]
    Tiles = []
    for tiling in range(numtilings):
        tilingX2 = tiling * 2
        coords = [tiling]
        b = tiling
        for q in qfloats:
            coords.append((q + b) // numtilings)
            b += tilingX2
        coords.extend(ints)
        # print("coords", coords)
        Tiles.append(hashcoords(coords, ihtORsize, readonly))
    return Tiles

