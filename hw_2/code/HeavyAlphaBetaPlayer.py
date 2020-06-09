import time as tm
import copy
import random
from dataclasses import dataclass
from GeneralPlayer import State, GeneralPlayer
from AlphaBetaPlayer import AlphaBetaPlayer
import C_CONSTANTS



class HeavyAlphaBetaPlayer(AlphaBetaPlayer):
    def __init__(self):
        super().__init__()
        self.agent_name = "HeavyAlphaBetaPlayer"        
        self.max_ground_distance = C_CONSTANTS.MAX_GROUND_DISTANCE_HEAVY  
