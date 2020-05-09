from typing import *
from dataclasses import dataclass
from enum import Enum

from framework import *
from .map_heuristics import AirDistHeuristic
from .cached_map_distance_finder import CachedMapDistanceFinder
from .mda_problem_input import *


__all__ = ['MDAState', 'MDACost', 'MDAProblem', 'MDAOptimizationObjective']


@dataclass(frozen=True)
class MDAState(GraphProblemState):
    """
    An instance of this class represents a state of MDA problem.
    This state includes:
        `current_site`:
            The current site where the ambulate is at.
            The initial state stored in this field the initial ambulance location (which is a `Junction` object).
            Other states stores the last visited reported apartment (object of type `ApartmentWithSymptomsReport`),
             or the last visited laboratory (object of type `Laboratory`).
        `tests_on_ambulance`:
            Stores the reported-apartments (objects of type `ApartmentWithSymptomsReport`) which had been visited,
             and their tests are still stored on the ambulance (hasn't been transferred to a laboratory yet).
        `tests_transferred_to_lab`:
            Stores the reported-apartments (objects of type `ApartmentWithSymptomsReport`) which had been visited,
             and their tests had already been transferred to a laboratory.
        `nr_matoshim_on_ambulance`:
            The number of matoshim currently stored on the ambulance.
            Whenever visiting a reported apartment, this number is decreased by the #roommates in this apartment.
            Whenever visiting a laboratory for the first time, we transfer the available matoshim from this lab
             to the ambulance.
        `visited_labs`:
            Stores the laboratories (objects of type `Laboratory`) that had been visited at least once.
    """

    current_site: Union[Junction, Laboratory, ApartmentWithSymptomsReport]
    tests_on_ambulance: FrozenSet[ApartmentWithSymptomsReport]
    tests_transferred_to_lab: FrozenSet[ApartmentWithSymptomsReport]
    nr_matoshim_on_ambulance: int
    visited_labs: FrozenSet[Laboratory]

    @property
    def current_location(self):
        if isinstance(self.current_site, ApartmentWithSymptomsReport) or isinstance(self.current_site, Laboratory):
            return self.current_site.location
        assert isinstance(self.current_site, Junction)
        return self.current_site

    def get_current_location_short_description(self) -> str:
        if isinstance(self.current_site, ApartmentWithSymptomsReport):
            return f'test @ {self.current_site.reporter_name}'
        if isinstance(self.current_site, Laboratory):
            return f'lab {self.current_site.name}'
        return 'initial-location'

    def __str__(self):
        return f'(' \
               f'loc: {self.get_current_location_short_description()} ' \
               f'tests on ambulance: ' \
               f'{[f"{reported_apartment.reporter_name} ({reported_apartment.nr_roommates})" for reported_apartment in self.tests_on_ambulance]} ' \
               f'tests transferred to lab: ' \
               f'{[f"{reported_apartment.reporter_name} ({reported_apartment.nr_roommates})" for reported_apartment in self.tests_transferred_to_lab]} ' \
               f'#matoshim: {self.nr_matoshim_on_ambulance} ' \
               f'visited labs: {[lab.name for lab in self.visited_labs]}' \
               f')'

    def __eq__(self, other):
        """
        This method is used to determine whether two given state objects represent the same state.
        """
        assert isinstance(other, MDAState)

        # [Ex.13]: Complete the implementation of this method!
        #  Note that you can simply compare two instances of `Junction` type
        #   (using equals `==` operator) because the class `Junction` explicitly
        #   implements the `__eq__()` method. The types `frozenset`, `ApartmentWithSymptomsReport`, `Laboratory`
        #   are also comparable (in the same manner).

        # can be implemented using hash??
        # if hash(self) == hash(other):
        #     return True

        # return False

        
        if self.current_site != other.current_site:
            return False

        if self.tests_on_ambulance != other.tests_on_ambulance:
            return False
         
        if self.tests_transferred_to_lab != other.tests_transferred_to_lab:
            return False

        if self.nr_matoshim_on_ambulance != other.nr_matoshim_on_ambulance:
            return False

        if self.visited_labs != other.visited_labs:
            return False




    def __hash__(self):
        """
        This method is used to create a hash of a state instance.
        The hash of a state being is used whenever the state is stored as a key in a dictionary
         or as an item in a set.
        It is critical that two objects representing the same state would have the same hash!
        """
        return hash((self.current_site, self.tests_on_ambulance, self.tests_transferred_to_lab,
                     self.nr_matoshim_on_ambulance, self.visited_labs))

    def get_total_nr_tests_taken_and_stored_on_ambulance(self) -> int:
        """
        This method returns the total number of of tests that are stored on the ambulance in this state.
         [Ex.13]: Implement this method.
         Notice that this method can be implemented using a single line of code - do so!
         Use python's built-it `sum()` function.
         Notice that `sum()` can receive an *ITERATOR* as argument; That is, you can simply write something like this:
        >>> sum(<some expression using item> for item in some_collection_of_items)
        """

        # tests_on_ambulance isa  set of ApartmentWithSymptomsReport
        # ApartmentWithSymptomsReport.nr_roommates shows the number of roommates


        return sum(test_taken.nr_roommates for test_taken in self.tests_on_ambulance)

        

        



class MDAOptimizationObjective(Enum):
    Distance = 'Distance'
    TestsTravelDistance = 'TestsTravelDistance'


@dataclass(frozen=True)
class MDACost(ExtendedCost):
    """
    An instance of this class is returned as an operator cost by the method
     `MDAProblem.expand_state_with_costs()`.
    The `SearchNode`s that will be created during the run of the search algorithm are going
     to have instances of `MDACost` in SearchNode's `cost` field (instead of float values).
    The reason for using a custom type for the cost (instead of just using a `float` scalar),
     is because we want the cumulative cost (of each search node and particularly of the final
     node of the solution) to be consisted of 2 objectives: (i) distance, and (ii) tests-travel.
    The field `optimization_objective` controls the objective of the problem (the cost we want
     the solver to minimize). In order to tell the solver which is the objective to optimize,
     we have the `get_g_cost()` method, which returns a single `float` scalar which is only the
     cost to optimize.
    This way, whenever we get a solution, we can inspect the 2 different costs of that solution,
     even though the objective was only one of the costs.
    Having said that, note that during this assignment we will mostly use the distance objective.
    """
    distance_cost: float = 0.0
    tests_travel_distance_cost: float = 0.0
    optimization_objective: MDAOptimizationObjective = MDAOptimizationObjective.Distance

    def __add__(self, other):
        assert isinstance(other, MDACost)
        assert other.optimization_objective == self.optimization_objective
        return MDACost(
            optimization_objective=self.optimization_objective,
            distance_cost=self.distance_cost + other.distance_cost,
            tests_travel_distance_cost=self.tests_travel_distance_cost + other.tests_travel_distance_cost)

    def get_g_cost(self) -> float:
        if self.optimization_objective == MDAOptimizationObjective.Distance:
            return self.distance_cost
        assert self.optimization_objective == MDAOptimizationObjective.TestsTravelDistance
        return self.tests_travel_distance_cost

    def __repr__(self):
        return f'MDACost(' \
               f'dist={self.distance_cost:11.3f}m, ' \
               f'tests-travel={self.tests_travel_distance_cost:11.3f}m)'


class MDAProblem(GraphProblem):
    """
    An instance of this class represents an MDA problem.
    """

    name = 'MDA'

    def __init__(self,
                 problem_input: MDAProblemInput,
                 streets_map: StreetsMap,
                 optimization_objective: MDAOptimizationObjective = MDAOptimizationObjective.Distance):
        self.name += f'({problem_input.input_name}({len(problem_input.reported_apartments)}):{optimization_objective.name})'
        initial_state = MDAState(
            current_site=problem_input.ambulance.initial_location,
            tests_on_ambulance=frozenset(),
            tests_transferred_to_lab=frozenset(),
            nr_matoshim_on_ambulance=problem_input.ambulance.initial_nr_matoshim,
            visited_labs=frozenset())
        super(MDAProblem, self).__init__(initial_state)
        self.problem_input = problem_input
        self.streets_map = streets_map
        self.map_distance_finder = CachedMapDistanceFinder(
            streets_map, AStar(AirDistHeuristic))
        self.optimization_objective = optimization_objective

    def expand_state_with_costs(self, state_to_expand: GraphProblemState) -> Iterator[OperatorResult]:
        """
        [Ex.13]: Implement this method!
        This method represents the `Succ: S -> P(S)` function of the MDA problem.
        The `Succ` function is defined by the problem operators as shown in class.
        The MDA problem operators are defined in the assignment instructions.
        It receives a state and iterates over its successor states.
        Notice that this its return type is an *Iterator*. It means that this function is not
         a regular function, but a `generator function`. Hence, it should be implemented using
         the `yield` statement.
        For each successor, an object of type `OperatorResult` is yielded. This object describes the
            successor state, the cost of the applied operator and its name. Look for its definition
            and use the correct fields in its c'tor. The operator name should be in the following
            format: `visit ReporterName` (with the correct reporter name) if an reported-apartment
            visit operator was applied (to take tests from the roommates of an apartment), or
            `go to lab LabName` if a laboratory visit operator was applied.
            The apartment-report object stores its reporter-name in one of its fields.
        Things you might want to use:
            - The method `self.get_total_nr_tests_taken_and_stored_on_ambulance()`.
            - The field `self.problem_input.laboratories`.
            - The field `self.problem_input.ambulance.taken_tests_storage_capacity`.
            - The method `self.get_reported_apartments_waiting_to_visit()` here.
            - The method `self.get_operator_cost()`.
            - The c'tor for `AmbulanceState` to create the new successor state.
            - Python's built-in method `frozenset()` to create a new frozen set (for fields that
              expect this type) from another collection (set/list/tuple/iterator).
            - Other fields of the state and the problem input.
            - Python's sets union operation (`some_set_or_frozenset | some_other_set_or_frozenset`).
        """

        # state_to_expand.get_total_nr_tests_taken_and_stored_on_ambulance()    -> INT
        # self.problem_input.laboratories                                       -> tuple
        # self.problem_input.ambulance.taken_tests_storage_capacity             -> INT
        # self.get_operator_cost()                                              -> MDACost
        # self.get_reported_apartments_waiting_to_visit(state_to_expand)        -> SET    those don't include taken and transfered


        assert isinstance(state_to_expand, MDAState)
        
        # check all the APARTMENT VISITS


        for apartment_to_visit in self.get_reported_apartments_waiting_to_visit(state_to_expand):

            # has enough space in ambulance
            space_on_ambulance = self.problem_input.ambulance.taken_tests_storage_capacity - state_to_expand.get_total_nr_tests_taken_and_stored_on_ambulance()
            if space_on_ambulance < apartment_to_visit.nr_roommates:
                continue ; # not enough space

            # check there's enough matoshim
            if state_to_expand.nr_matoshim_on_ambulance < apartment_to_visit.nr_roommates:
                continue ; # not enought matoshim

            # we can visit the state. Calculate the cost

            successor_state =  MDAState(
                            current_site=apartment_to_visit,
                            tests_on_ambulance=(state_to_expand.tests_on_ambulance| frozenset({apartment_to_visit})),
                            tests_transferred_to_lab=state_to_expand.tests_transferred_to_lab,
                            nr_matoshim_on_ambulance=state_to_expand.nr_matoshim_on_ambulance - apartment_to_visit.nr_roommates,
                            visited_labs=state_to_expand.visited_labs)

            operator_cost = self.get_operator_cost(state_to_expand, successor_state)
            reporter_name = apartment_to_visit.reporter_name
            op_result = OperatorResult(successor_state, operator_cost, f"Visit {reporter_name}")

            yield op_result
            


        # check all the LAB VISITS

        not_visited_labs = set(self.problem_input.laboratories) - state_to_expand.visited_labs

        for lab_to_visit in not_visited_labs:

            # check refrigerator is not empty
            if len(state_to_expand.tests_on_ambulance) ==  0 :
                continue ; # refrigerator is not empty

            # we can visit the state. Calculate the cost

            successor_state =  MDAState(
                            current_site=lab_to_visit,
                            tests_on_ambulance=frozenset({}),
                            tests_transferred_to_lab=( state_to_expand.tests_transferred_to_lab | state_to_expand.tests_on_ambulance),
                            nr_matoshim_on_ambulance=state_to_expand.nr_matoshim_on_ambulance + lab_to_visit.max_nr_matoshim,
                            visited_labs=(state_to_expand.visited_labs | frozenset({lab_to_visit})))

            operator_cost = self.get_operator_cost(state_to_expand, successor_state)

            LabName = lab_to_visit.name
            op_result = OperatorResult(successor_state, operator_cost, f"go to lab {LabName}")

            yield op_result

    def get_operator_cost(self, prev_state: MDAState, succ_state: MDAState) -> MDACost:
        """
        Calculates the operator cost (of type `MDACost`) of an operator (moving from the `prev_state`
         to the `succ_state`. The `MDACost` type is defined above in this file (with explanations).
        Use the formal MDA problem's operator costs definition presented in the assignment-instructions.
        [Ex.13]: implement this method!
        Use the method `self.map_distance_finder.get_map_cost_between()` to calculate the distance
         between to junctions.
        """

        # calculate the cost between 2 states

        cost_between_nodes = self.map_distance_finder.get_map_cost_between(src_junction= prev_state.current_location, 
                                                                            tgt_junction=succ_state.current_location)
        
        # physical distance
        distance_cost = cost_between_nodes

        # get the number of tests on the ambulance
        num_tests_on_ambulance = 0

        for apartWithSymptomsReport in prev_state.tests_on_ambulance:
            num_tests_on_ambulance += apartWithSymptomsReport.nr_roommates

        tests_travel_distance_cost = distance_cost * num_tests_on_ambulance

              
        mda_cost_obj = MDACost(distance_cost =  cost_between_nodes, 
                                tests_travel_distance_cost = tests_travel_distance_cost, 
                                optimization_objective= self.optimization_objective)

        return mda_cost_obj

    def is_goal(self, state: GraphProblemState) -> bool:
        """
        This method receives a state and returns whether this state is a goal.
        [Ex.13]: implement this method using a single `return` line!
         Use sets/frozensets comparison (`some_set == some_other_set`).
         In order to create a set from some other collection (list/tuple) you can just `set(some_other_collection)`.
        """
        assert isinstance(state, MDAState)


        original_set = frozenset(self.problem_input.reported_apartments)
        if (original_set == (state.tests_on_ambulance | state.tests_transferred_to_lab )) :
            return True
        return False



    def get_zero_cost(self) -> Cost:
        """
        Overridden method of base class `GraphProblem`. For more information, read
         documentation in the default implementation of this method there.
        In this problem the accumulated cost is not a single float scalar, but an
         extended cost, which actually includes 2 scalar costs.
        """
        return MDACost(optimization_objective=self.optimization_objective)

    def get_reported_apartments_waiting_to_visit(self, state: MDAState) -> Set[ApartmentWithSymptomsReport]:
        """
        This method returns a set of all reported-apartments that haven't been visited yet.
        [Ex.13]: Implement this method.
            Use sets difference operation (`some_set - some_other_set`).
            Note: Given a collection of items, you can create a new set of these items simply by
                `set(my_collection_of_items)`. Then you can use set operations over this newly
                generated set.
            Note: This method can be implemented using a single line of code. Try to do so.
        """

        # from the total number of reported apartments
        # reduce the set of visited and still in ambulance
        # reduce the set of visited and already in lab

        original_set = set(self.problem_input.reported_apartments)
        return original_set - state.tests_on_ambulance - state.tests_transferred_to_lab



    def get_all_certain_junctions_in_remaining_ambulance_path(self, state: MDAState) -> List[Junction]:
        """
        This method returns a list of junctions that are part of the remaining route of the ambulance.
        This includes the ambulance's current location, and the locations of the reported apartments
         that hasn't been visited yet.
        The list should be ordered by the junctions index ascendingly (small to big).
        TODO [Ex.16]: Implement this method.
            Use the method `self.get_reported_apartments_waiting_to_visit(state)`.
            Use python's `sorted(..., key=...)` function.
        """
        raise NotImplementedError  # TODO: remove this line!
