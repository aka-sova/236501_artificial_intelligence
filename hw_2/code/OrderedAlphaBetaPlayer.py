import time as tm
import copy
import random 
import C_CONSTANTS

from dataclasses import dataclass
from GeneralPlayer import State, GeneralPlayer



class OrderedAlphaBetaPlayer(GeneralPlayer):
    def __init__(self):
        super().__init__()

        self.agent_name = "OrderedAlphaBetaPlayer"
        self.state = None        
        self.directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

        self.leaves_developed = 0
        self.heuristics_used = 0
        self.branches_pruned = 0

        self.lastMinimaxValues = {}

    def make_move(self, time_limit) -> tuple:  # time parameter is not used, we assume we have enough time.

        """Implement the RB - MiniMax algorithm with iterative deepening"""

        ID_start_time  = tm.time()

        prev_loc = self.state.my_loc
        self.state.board[prev_loc] = -1
        self.move_number += 1

        self.lastMinimaxValues = {}


        # time = 1000

        # init
        current_depth = 1

        
        # DEPTH = 1
        # print(f"Depth : {current_depth}")

        self.leaves_developed = 0
        (best_new_move, max_value ) = self.rb_ordered_alphabeta(self.state, DecidingAgent = "Me",\
             D = current_depth, Alpha = float('-inf'), Beta = float('inf'), isRoot = True)
        best_move_so_far = best_new_move

        # print(f"Move value : {max_value}")
        # print(f"Leaves developed: {self.leaves_developed}, Heuristics used : {self.heuristics_used}, Branches pruned: {self.branches_pruned}")

        time_until_now = tm.time() - ID_start_time

        next_iteration_max_time = self.predict_next_iteration(time_until_now) # time_until_now = last_iteration_time
    
        max_depth = self.state.while_tiles_am

        while time_until_now + next_iteration_max_time < time_limit  and current_depth < max_depth:

            current_depth += 1

            # perform the next depth iteration  
            iteration_start_time = tm.time()

            # print(f"Depth : {current_depth}")

            self.leaves_developed = 0
            self.branches_pruned = 0
            self.heuristics_used = 0

            (best_new_move, max_value ) = self.rb_ordered_alphabeta(self.state, DecidingAgent = "Me", \
                D = current_depth, Alpha = float('-inf'), Beta = float('inf'),\
                    isRoot=True)
            best_move_so_far = best_new_move

            # print(f"Move value : {max_value}")

            if max_value == -99999 or max_value == 99999:
                # the only outcome is losing or winning
                break

            last_iteration_time = tm.time() - iteration_start_time

            # print(f"Leaves developed: {self.leaves_developed}, Heuristics used : {self.heuristics_used}, Branches pruned: {self.branches_pruned}")
            # print(f"Predicted time : {next_iteration_max_time}, time elapsed: {last_iteration_time}")

            

            next_iteration_max_time = self.predict_next_iteration(last_iteration_time)
            time_until_now = tm.time() - ID_start_time

        # print("====================")
        # print(f"Agent: {self.agent_name}")
        # print(f"Move No: {self.move_number}")
        # print(f"Depth reached : {current_depth}")
        # print(f"Leaves developed: {self.leaves_developed}, Heuristics used : {self.heuristics_used}, Branches pruned: {self.branches_pruned}")
        # print(f"Move chosen: {best_move_so_far}  Value = {max_value}")
        # print("====================")

        self.state.update(best_move_so_far, "Me")

        # if C_CONSTANTS.USE_COMPARISON:
        #     return current_depth, max_value
        return best_move_so_far




    def rb_ordered_alphabeta(self, CurrentState : State,  DecidingAgent : str, D : int, Alpha: int, Beta: int, isRoot : bool) -> tuple:
        """Get the next most beneficial move

        CurrentState = TUPLE(BOARD, MY_LOC, ENEMY_LOC)
        
        DecidingAgent = Me / Enemy
        State         = tuple(i,j)
        D             = int(depth)
        """


        # check if final state, then return the Utility of the state
        if self.is_final_state(CurrentState, DecidingAgent):
            state_utility = self.get_utility(CurrentState, DecidingAgent)
            self.leaves_developed +=1
            return (None, state_utility)


        if D == 0:
            heuristic_value = self.get_heuristic_value(CurrentState, DecidingAgent)
            self.heuristics_used += 1
            self.leaves_developed +=1
            return (None, heuristic_value)

        # get all the children states
        children_moves = self.get_children(CurrentState, DecidingAgent)
        random.shuffle(children_moves)

        if isRoot:  
            # reorder the moves according to the lastMinimaxValues
            if self.lastMinimaxValues != {}:
                ordered_children_moves_dict = dict(sorted(self.lastMinimaxValues.items(), key = lambda kv:(kv[1])))
                children_moves = list(ordered_children_moves_dict.keys())
                # this part will be open ONLY by the ME agent. so we use the MAX
                children_moves.reverse() # we want to open the moves with BIGGERST minimax value first 

        
        if DecidingAgent == "Me":
            # MAX

            CurMax = float('-inf')
            CurMaxMove = None

            for child_move in children_moves:

                # update the state (board, locations) after a child move is made
                ChildState = copy.deepcopy(CurrentState)
                ChildState.update(child_move, DecidingAgent)
                
                
                _, max_child_value = self.rb_ordered_alphabeta(ChildState, "Enemy", D-1, Alpha, Beta, False)
                # print(f"D = {D} , max_child_value =  {str(max_child_value)}")
                if max_child_value > CurMax:
                    CurMax = max_child_value
                    CurMaxMove = child_move

                # ALPHABETA
                Alpha = max(CurMax, Alpha)
                if CurMax >= Beta:
                    self.branches_pruned += 1
                    return (None, float('inf'))

                if isRoot:
                    #  root will always open all its children
                    # will always start with Me
                    self.lastMinimaxValues[child_move] = max_child_value
            
            return (CurMaxMove, CurMax)


        else:
            # Enemy agent - MIN

            CurMin = float("inf")
            CurMinMove = None

            for child_move in children_moves:
                
                # update the state (board, locations) after a child move is made
                ChildState = copy.deepcopy(CurrentState)
                ChildState.update(child_move, DecidingAgent)

                
                _, min_child_value = self.rb_ordered_alphabeta(ChildState, "Me", D-1, Alpha, Beta, False)
                if min_child_value < CurMin:
                    CurMin = min_child_value
                    CurMinMove = child_move

                # ALPHABETA
                Beta = min(CurMin, Beta)
                if CurMin <= Alpha:
                    self.branches_pruned += 1
                    return (None, float('-inf'))
            
            return (CurMinMove, CurMin)