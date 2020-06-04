import time as tm
import copy

from dataclasses import dataclass
from MaxGroundHeuristic import *


@dataclass
class State:
    """Board is already updates, my_loc and enemy_loc simply indicate the locations"""
    board: list
    my_loc: tuple
    enemy_loc: tuple

    def update(self, move : [], DecidingAgent):
        """ update the current board and locations based on the move"""

        if DecidingAgent == "Me":
            new_loc = (self.my_loc[0] + move[0], self.my_loc[1] + move[1])
            self.my_loc = new_loc
            
        else:
            new_loc = (self.enemy_loc[0] + move[0], self.enemy_loc[1] + move[1])
            self.enemy_loc = new_loc

        self.board[new_loc] = -1

    def __hash__(self):
        return id(self)
        



class GeneralPlayer:
    def __init__(self):
        self.state = None        
        self.directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

        self.leaves_developed = 0
        self.heuristics_used = 0

        self.max_ground_heuristic_func = MaxGroudHeuristic()



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

        self.state = State(board, my_loc, enemy_loc)
        self.state.board[enemy_loc] = -1

    def max_ground_value(self, state : State, DecidingAgent : str):
        """ Return the value based on MaxGroundHeuristic"""

        self.max_ground_heuristic_func.board = state.board
        self.max_ground_heuristic_func.player_loc = state.my_loc
        self.max_ground_heuristic_func.opp_loc = state.enemy_loc

        ground_value = self.max_ground_heuristic_func.evaluate()
        
        return ground_value


    def state_score(self, state : State, DecidingAgent : str):
        """Return the numer of available states from a certain location

        0 = all moves available
        1 = 3 moves available
        2 = 2 moves available
        3 = 1 moves available
        -1 = no moves available
        
        """

        if DecidingAgent == "Me":
            board, loc = state.board, state.my_loc
        else:
            board, loc = state.board, state.enemy_loc

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
        heuristic_variables.append(self.state_score(state, DecidingAgent))

        # TODO add heuristics
        heuristic_variables.append(self.max_ground_value(state, DecidingAgent))



        return sum(heuristic_variables)


    def predict_next_iteration(self, last_elapsed_time):

        # b = 3 in the worst case
        
        b = 3
        new_leaves_estimation = self.leaves_developed * b

        new_time = last_elapsed_time * (new_leaves_estimation / self.leaves_developed)

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
                utility = -100 # enemy wins
            else:
                utility = 0  # teko

        else:
            # ENEMY is in final state and has no moves

            # if I have no moves in the next turn - it's a tie
            # if I still have moves - victory!

            # get successors of "Me"
            NextDecidingAgent = "Me"
            children_moves = self.get_children(State, NextDecidingAgent)

            if len(children_moves) > 0:
                utility = 100  # win
            else:
                utility = 0  # teko

        # find a better number according to heuristics

        # If DecidingAgent wins on this turn, return 100
        # Else return -100

        # utility = None

        # if DecidingAgent == "Me":
        #     # ME got into final state and have no moves - bad
        #     utility = -100
        # else:
        #     # ENEMY is in final state and has no moves - good.
        #     utility = 100
        

        return utility

    def set_rival_move(self, loc):
        """adjust board after rival made a move"""
        self.state.board[loc] = -1
        self.state.enemy_loc = loc
