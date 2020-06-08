
import math 

class EuclideanDistanceHeuristic:
    def __init__(self):
        self.name = 'EuclideanDistanceHeuristic'

    def evaluate(self, player_loc=None, opp_loc=None):
        """ Calculate the Euclidean distance between the player and the opponent"""

        distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(player_loc, opp_loc)]))

        return distance


