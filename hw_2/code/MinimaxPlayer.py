import time as tm
import copy

from dataclasses import dataclass


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
        



class MinimaxPlayer:
    def __init__(self):
        self.state = None        
        self.directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

        self.leaves_developed = 0
        self.heuristics_used = 0



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

    def count_ones(self, board):
        """Counts the location of the agent tile (1)"""
        counter = 0
        for i, row in enumerate(board):
            for j, val in enumerate(row):
                if val == 1:
                    counter += 1
        return counter

    def get_heuristic_value(self, state : State, DecidingAgent : str):
        """Will return the heuristic of a specific state"""

        # heuristic will consist of numerous variables

        heuristic_variables = []

        # num of available states from a certain location
        heuristic_variables.append(self.state_score(state, DecidingAgent))

        # TODO add heuristics

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

        # if all_occupied == True:
        #     print('asdsad')

        return all_occupied

    def get_utility(self, State : State, DecidingAgent : str):

        """Setting some critical values for winning or losing        """

        # find a better number according to heuristics

        # If DecidingAgent wins on this turn, return 100
        # Else return -100

        utility = None

        if DecidingAgent == "Me":
            # ME got into final state and have no moves - bad
            utility = -100
        else:
            # ENEMY is in final state and has no moves - good.
            utility = 100
        

        return utility


    def make_move(self, time_limit) -> tuple:  # time parameter is not used, we assume we have enough time.

        """Implement the RB - MiniMax algorithm with iterative deepening"""

        ID_start_time  = tm.time()
        # assert self.count_ones(self.state.board) == 1

        prev_loc = self.state.my_loc
        self.state.board[prev_loc] = -1

        # assert self.count_ones(self.state.board) == 0


        # time = 1000

        # init
        current_depth = 1
        next_iteration_max_time = 0
        self.heuristics_used = 1 # > 0

        
        # DEPTH = 1
        print(f"Depth : {current_depth}")
        self.leaves_developed = 0
        (best_new_move, max_value ) = self.rb_minimax(self.state, DecidingAgent = "Me", D = current_depth)
        best_move_so_far = best_new_move

        time_until_now = tm.time() - ID_start_time
    

        while time_until_now + next_iteration_max_time < time_limit:
            # perform the next depth iteration  
            iteration_start_time = tm.time()

            print(f"Depth : {current_depth}")
            self.leaves_developed = 0
            (best_new_move, max_value ) = self.rb_minimax(self.state, DecidingAgent = "Me", D = current_depth)
            best_move_so_far = best_new_move

            if max_value == -100 or max_value == 100:
                # the only outcome is losing or winning
                break

            last_iteration_time = tm.time() - iteration_start_time

            print(f"Leaves developed: {self.leaves_developed}, Heuristics used : {self.heuristics_used}")
            print(f"Predicted time : {next_iteration_max_time}, time elapsed: {last_iteration_time}")

            current_depth += 1

            next_iteration_max_time = self.predict_next_iteration(last_iteration_time)
            time_until_now = tm.time() - ID_start_time



        print(f"Move chosen: {best_move_so_far}")

        self.state.update(best_move_so_far, "Me")


        return best_move_so_far



    def rb_minimax(self, CurrentState : State,  DecidingAgent : str, D : int) -> tuple:
        """Get the next most beneficial move

        CurrentState = TUPLE(BOARD, MY_LOC, ENEMY_LOC)
        
        DecidingAgent = Me / Enemy
        State         = tuple(i,j)
        D             = int(depth)
        """


        # check if final state, then return the Utility of the state
        if self.is_final_state(CurrentState, DecidingAgent):
            state_utility = self.get_utility(CurrentState, DecidingAgent)

            return (None, state_utility)


        if D == 0:
            heuristic_value = self.get_heuristic_value(CurrentState, DecidingAgent)
            self.heuristics_used += 1
            return (None, heuristic_value)

        # get all the children states
        children_moves = self.get_children(CurrentState, DecidingAgent)
        
        if DecidingAgent == "Me":
            # MAX

            CurMax = float('-inf')          # check which value to put
            CurMaxMove = None

            for child_move in children_moves:

                # update the state (board, locations) after a child move is made
                ChildState = copy.deepcopy(CurrentState)
                ChildState.update(child_move, DecidingAgent)
                
                self.leaves_developed +=1
                _, max_child_value = self.rb_minimax(ChildState, "Enemy", D-1)
                if max_child_value > CurMax:
                    CurMax = max_child_value
                    CurMaxMove = child_move
            
            return (CurMaxMove, CurMax)


        else:
            # Enemy agent - MIN

            CurMin = float("inf")          # check which value to put
            CurMinMove = None

            for child_move in children_moves:
                
                # update the state (board, locations) after a child move is made
                ChildState = copy.deepcopy(CurrentState)
                ChildState.update(child_move, DecidingAgent)

                self.leaves_developed +=1
                _, min_child_value = self.rb_minimax(ChildState, "Me", D-1)
                if min_child_value < CurMin:
                    CurMin = min_child_value
                    CurMinMove = child_move
            
            return (CurMinMove, CurMin)

    def set_rival_move(self, loc):
        """adjust board after rival made a move"""
        self.state.board[loc] = -1
        self.state.enemy_loc = loc
