import time as tm
import copy
import numpy as np
from dataclasses import dataclass
from MaxGroundHeuristic import *
from DistFromOpponentHeuristic import *
from EuclideanDistanceHeuristic import *


@dataclass
class State:
    """Board is already updates, my_loc and enemy_loc simply indicate the locations"""
    board: list
    my_loc: tuple
    enemy_loc: tuple
    # free_tiles_ratio : float
    grey_tiles_am : int

    def update(self, move : [], DecidingAgent):
        """ update the current board and locations based on the move"""

        if DecidingAgent == "Me":
            new_loc = (self.my_loc[0] + move[0], self.my_loc[1] + move[1])
            self.my_loc = new_loc
            
        else:
            new_loc = (self.enemy_loc[0] + move[0], self.enemy_loc[1] + move[1])
            self.enemy_loc = new_loc

        self.board[new_loc] = -1

        self.grey_tiles_am += 1

    @property
    def free_tiles_ratio(self):
        global board_size
        return round(self.grey_tiles_am / board_size,2)

    def __hash__(self):
        return id(self)
        



class GeneralPlayer:
    def __init__(self):
        self.state = None        
        self.directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

        self.leaves_developed = 0
        self.heuristics_used = 0

        self.max_ground_heuristic_func = MaxGroundHeuristic()
        self.dist_from_opponent_heuristic = DistFromOpponentHeuristic()
        self.euclidean_distance_heuristic = EuclideanDistanceHeuristic()

        self.heuristic_function_type = None


    def set_game_params(self, board):

               
        for i, row in enumerate(board):
            for j, val in enumerate(row):
                if val == 1:
                    my_loc = (i, j)
                    break

        for i, row in enumerate(board):
            for j, val in enumerate(row):
                if val == 2:
                    enemy_loc = (i, j)
                    break

        global board_size
        board_size = board.shape[0] * board.shape[1]

        unique, counts = np.unique(board, return_counts=True)
        counts_dict = dict(zip(unique, counts))

        self.state = State(board, my_loc, enemy_loc, counts_dict[-1])
        

        self.state.board[enemy_loc] = -1
        self.state.board[my_loc] = -1
        self.state.grey_tiles_am += 2



        # calculate the ratio of while tiles to the grey tiles
    def max_ground_value(self, state : State):
        """ Return the value based on MaxGroundHeuristic"""

        self.max_ground_heuristic_func.board = state.board
        self.max_ground_heuristic_func.player_loc = state.my_loc
        self.max_ground_heuristic_func.opp_loc = state.enemy_loc

        ground_value = self.max_ground_heuristic_func.evaluate()
        
        return ground_value

    def distance_from_opponent(self, state : State):
        """ Return the current distance to the opponent"""

        self.dist_from_opponent_heuristic.board = state.board
        self.dist_from_opponent_heuristic.player_loc = state.my_loc
        self.dist_from_opponent_heuristic.opp_loc = state.enemy_loc

        distance_from_opp = self.dist_from_opponent_heuristic.evaluate()

        return distance_from_opp

    def euclidean_distance_from_opponent(self, state : State):
        

        euclidean_distance_from_opp = self.euclidean_distance_heuristic.evaluate(state.my_loc, state.enemy_loc)

        return euclidean_distance_from_opp

    def state_score(self, state : State):
        """Return the numer of available states from a certain location

        0 = all moves available
        1 = 3 moves available
        2 = 2 moves available
        3 = 1 moves available
        -1 = no moves available
        
        """

        # always estimate the state score of 'Me'
        board, loc = state.board, state.my_loc


        # if DecidingAgent == "Me":
        #     board, loc = state.board, state.my_loc
        # else:
        #     board, loc = state.board, state.enemy_loc

        num_steps_available = 0
        for d in self.directions:
            i = loc[0] + d[0]
            j = loc[1] + d[1]
            if 0 <= i < len(board) and 0 <= j < len(board[0]) and board[i][j] == 0:  # then move is legal
                num_steps_available += 1

        if num_steps_available == 0:
            return -1
        else:
            # return 4 - num_steps_available
            return num_steps_available

    def get_heuristic_value(self, state : State, DecidingAgent : str):
        """Will return the heuristic of a specific state"""

        # heuristic will consist of numerous variables

        heuristic_variables = []

        # num of available states from a certain location
        heuristic_variables.append(self.state_score(state))

        # calculate the territory advantage of the player
        heuristic_variables.append(self.max_ground_value(state))

        # calculate the distance from an opponent
        dist_from_opp = self.distance_from_opponent(state)
        if dist_from_opp != -1:
            # if not - meaning we never meet an opponent and should only care about maximizing our own territory
            heuristic_variables.append(dist_from_opp)

        # calculate the Euclidean distance from an opponent
        # heuristic_variables.append(self.euclidean_distance_from_opponent(state))



        return sum(heuristic_variables)


    def predict_next_iteration(self, last_elapsed_time):
        
        # b = 3
        # new_leaves_estimation = self.leaves_developed * b

        # new_time = last_elapsed_time * (new_leaves_estimation / self.leaves_developed)

        new_time = last_elapsed_time * 4

        return new_time




    def get_children(self, state : State, DecidingAgent : str):
        """return the list consisting of all children from a given state"""

        children = []

        if DecidingAgent == "Me":
            for d in self.directions:
                i = state.my_loc[0] + d[0]
                j = state.my_loc[1] + d[1]

                if 0 <= i < len(state.board) and 0 <= j < len(state.board[0]) and state.board[i][j] == 0:  # then move is legal

                    new_loc = (i, j)
                    if state.board[new_loc] == 0:
                        children.append(d)
        else:
            for d in self.directions:
                i = state.enemy_loc[0] + d[0]
                j = state.enemy_loc[1] + d[1]

                if 0 <= i < len(state.board) and 0 <= j < len(state.board[0]) and state.board[i][j] == 0:  # then move is legal

                    new_loc = (i, j)
                    if state.board[new_loc] == 0:
                        children.append(d)

        return children

    def is_final_state(self, state : State, DecidingAgent: str ):

        """ The state is final if the game will end on this turn or the next turn"""

        # check if ends on this turn. Meaning all neighbor tiles are occupied 
        # loop on all the directions from current tile

        if DecidingAgent == "Me":

            all_occupied = True

            for d in self.directions:
                i = state.my_loc[0] + d[0]
                j = state.my_loc[1] + d[1]

                if 0 <= i < len(state.board) and 0 <= j < len(state.board[0]):  # then tile is legal
                    if state.board[i][j] not in [-1, 2]: 
                        # if not gray tile or enemy tile inside - meaning we have another tile to go to
                        all_occupied = False
                        break

        else:

            all_occupied = True

            for d in self.directions:
                i = state.enemy_loc[0] + d[0]
                j = state.enemy_loc[1] + d[1]

                if 0 <= i < len(state.board) and 0 <= j < len(state.board[0]):  # then tile is legal
                    if state.board[i][j] not in [-1, 2]: 
                        # if not gray tile or enemy tile inside - meaning we have another tile to go to
                        all_occupied = False
                        break        

        return all_occupied

    def get_utility(self, State : State, DecidingAgent : str):

        """Setting some critical values for winning or losing        """

        utility = None
        if DecidingAgent == "Me":
            # ME got into final state and have no moves. It's either loss or a tie

            NextDecidingAgent = "Enemy"
            children_moves = self.get_children(State, NextDecidingAgent)

            if len(children_moves) > 0:
                #utility = -100 # enemy wins
                utility = -99999 # enemy wins
            else:
                # utility = 0  # teko
                utility = float('-9999') # not as bad as losing

        else:
            # ENEMY is in final state and has no moves

            # if I have no moves in the next turn - it's a tie
            # if I still have moves - victory!

            # get successors of "Me"
            NextDecidingAgent = "Me"
            children_moves = self.get_children(State, NextDecidingAgent)

            if len(children_moves) > 0:
                utility = 99999  # win
            else:
                utility = 0  # teko
                utility = float('-9999')  # teko
        

        return utility

    def set_rival_move(self, loc):
        """adjust board after rival made a move"""
        self.state.board[loc] = -1
        self.state.enemy_loc = loc
