import time as tm
import copy

from dataclasses import dataclass
from GeneralPlayer import State, GeneralPlayer



class MinimaxPlayer(GeneralPlayer):
    def __init__(self):
        super().__init__()
        self.state = None        
        self.directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

        self.leaves_developed = 0
        self.heuristics_used = 0

    def make_move(self, time_limit) -> tuple:  # time parameter is not used, we assume we have enough time.

        """Implement the RB - MiniMax algorithm with iterative deepening"""

        ID_start_time  = tm.time()

        prev_loc = self.state.my_loc
        self.state.board[prev_loc] = -1


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

        print(f"Move value : {max_value}")
        print(f"Leaves developed: {self.leaves_developed}, Heuristics used : {self.heuristics_used}")

        time_until_now = tm.time() - ID_start_time
    

        while time_until_now + next_iteration_max_time < time_limit:
            # perform the next depth iteration  
            iteration_start_time = tm.time()

            print(f"Depth : {current_depth}")
            self.leaves_developed = 0
            (best_new_move, max_value ) = self.rb_minimax(self.state, DecidingAgent = "Me", D = current_depth)
            best_move_so_far = best_new_move

            print(f"Move value : {max_value}")

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
            self.leaves_developed +=1
            return (None, state_utility)


        if D == 0:
            heuristic_value = self.get_heuristic_value(CurrentState, DecidingAgent)
            self.heuristics_used += 1
            self.leaves_developed +=1
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
                
                
                _, max_child_value = self.rb_minimax(ChildState, "Enemy", D-1)
                # print(f"D = {D} , max_child_value =  {str(max_child_value)}")
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