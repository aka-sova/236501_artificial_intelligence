

""" define the constants that will be used in the algorithm """

MAX_GROUND_DISTANCE                 = 10  # define the radius to calculate the ground advantage over the opponent
MAX_GROUND_FACTOR                   = 1  # scale factor

DIST_FROM_OPP_RELEVANT              = 10  # below this value, the distance from opponent will not have any effect on the heuristics
                                          # because the max ground will already calculate
DIST_FROM_OPP_FACTOR                = -1  # scale factor  - is negative because we want to MINIMIZE the distance to the opponent