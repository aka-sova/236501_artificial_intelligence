from framework import *
from problems import *

from matplotlib import pyplot as plt
import numpy as np
from typing import List, Union, Optional

# Load the streets map
streets_map = StreetsMap.load_from_csv(Consts.get_data_file_path("tlv_streets_map.csv"))

# Make sure that the whole execution is deterministic.
# This is important, because we expect to get the exact same results
# in each execution.
Consts.set_seed()


# --------------------------------------------------------------------
# ------------------------ StreetsMap Problem ------------------------
# --------------------------------------------------------------------

def plot_distance_and_expanded_wrt_weight_figure(
        problem_name: str,
        weights: Union[np.ndarray, List[float]],
        total_cost: Union[np.ndarray, List[float]],
        total_nr_expanded: Union[np.ndarray, List[int]]):
    """
    Use `matplotlib` to generate a figure of the distance & #expanded-nodes
     w.r.t. the weight.
    [Ex.12]: Complete the implementation of this method.
    """
    weights, total_cost, total_nr_expanded = np.array(weights), np.array(total_cost), np.array(total_nr_expanded)
    assert len(weights) == len(total_cost) == len(total_nr_expanded)
    assert len(weights) > 0
    is_sorted = lambda a: np.all(a[:-1] <= a[1:])
    assert is_sorted(weights)

    fig, ax1 = plt.subplots()

    # : Plot the total distances with ax1. Use `ax1.plot(...)`.
    # : Make this curve colored blue with solid line style.
    # : Set its label to be 'Solution cost'.
    # See documentation here:
    # https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.plot.html
    # You can also Google for additional examples.


    p1, = ax1.plot(weights, total_cost, color='blue', linestyle='solid', linewidth = 2, label = "Solution cost") 

    # ax1: Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Solution cost', color='b')
    ax1.tick_params('y', colors='b')
    ax1.set_xlabel('weight')

    # Create another axis for the #expanded curve.
    ax2 = ax1.twinx()

    # Plot the total expanded with ax2. Use `ax2.plot(...)`.
    # Make this curve colored red with solid line style.
    # Set its label to be '#Expanded states'.

    p2, = ax2.plot(weights, total_nr_expanded, color='red', linestyle='solid', linewidth = 2, label = "#Expanded states") 


    # ax2: Make the y-axis label, ticks and tick labels match the line color.
    ax2.set_ylabel('#Expanded states', color='r')
    ax2.tick_params('y', colors='r')

    curves = [p1, p2]
    ax1.legend(curves, [curve.get_label() for curve in curves])

    fig.tight_layout()
    plt.title(f'Quality vs. time for wA* \non problem {problem_name}')
    plt.show()


def run_astar_for_weights_in_range(heuristic_type: HeuristicFunctionType, problem: GraphProblem, n: int = 30,
                                   max_nr_states_to_expand: Optional[int] = 30_000,
                                   low_heuristic_weight: float = 0.5, high_heuristic_weight: float = 0.95):
    # [Ex.12]:
    #  1. Create an array of `n` numbers equally spread in the segment
    #     [low_heuristic_weight, high_heuristic_weight]
    #     (including the edges). You can use `np.linspace()` for that.
    #  2. For each weight in that array run the wA* algorithm, with the
    #     given `heuristic_type` over the given problem. For each such run,
    #     if a solution has been found (res.is_solution_found), store the
    #     cost of the solution (res.solution_g_cost), the number of
    #     expanded states (res.nr_expanded_states), and the weight that
    #     has been used in this iteration. Store these in 3 lists (list
    #     for the costs, list for the #expanded and list for the weights).
    #     These lists should be of the same size when this operation ends.
    #     Don't forget to pass `max_nr_states_to_expand` to the AStar c'tor.
    #  3. Call the function `plot_distance_and_expanded_wrt_weight_figure()`
    #     with these 3 generated lists.

    weights_list = np.linspace(start = low_heuristic_weight, stop = high_heuristic_weight, num = n)
    g_cost_list         = []
    expanded_st_list    = []
    used_weights_list   = []

    for weight in weights_list:
        a_star_air_dist = AStar(heuristic_function_type=heuristic_type, 
                                heuristic_weight=weight,
                                max_nr_states_to_expand = 100000)
        res = a_star_air_dist.solve_problem(problem=problem)

        if res.is_solution_found:
            # print(f"\tFound! Weight = {round(weight,3)}, G_cost = {round(res.solution_g_cost,3)}, Expanded = {res.nr_expanded_states}")
            g_cost_list.append(res.solution_g_cost)
            expanded_st_list.append(int(res.nr_expanded_states))
            used_weights_list.append(weight)

    plot_distance_and_expanded_wrt_weight_figure(problem_name=problem.name,
                                                weights=used_weights_list,
                                                total_cost=g_cost_list,
                                                total_nr_expanded=expanded_st_list)


def toy_map_problem_experiments():
    print()
    print('Solve the map problem.')

    # Ex.8
    # Just run it and inspect the printed result.
    toy_map_problem = MapProblem(streets_map, 54, 549)

    # print("\n\nUNIFORM COST : \n")
    uc = UniformCost()
    res = uc.solve_problem(toy_map_problem)
    print(res)

    # Ex.10
    # create an instance of `AStar` with the `NullHeuristic`,
    #       solve the same `toy_map_problem` with it and print the results (as before).
    # Notice: AStar constructor receives the heuristic *type* (ex: `MyHeuristicClass`),
    #         and NOT an instance of the heuristic (eg: not `MyHeuristicClass()`).
    
    # print("\n\nA STAR  (Null Heuristic) : \n")
    a_star = AStar(heuristic_function_type=NullHeuristic)
    res = a_star.solve_problem(problem=toy_map_problem)
    print(res)


    # Ex.11
    # create an instance of `AStar` with the `AirDistHeuristic`,
    #       solve the same `toy_map_problem` with it and print the results (as before).
    
    # print("\n\nA STAR  (Air Distance Heuristic) : \n")
    a_star_air_dist = AStar(heuristic_function_type=AirDistHeuristic)
    res = a_star_air_dist.solve_problem(problem=toy_map_problem)
    print(res)

    # Ex.12
    # 
    #  1. Complete the implementation of the function
    #     `run_astar_for_weights_in_range()` (upper in this file).
    #  2. Complete the implementation of the function
    #     `plot_distance_and_expanded_wrt_weight_figure()`
    #     (upper in this file).
    #  3. Call here the function `run_astar_for_weights_in_range()`
    #     with `AirDistHeuristic` and `toy_map_problem`.

    # print("\n\nA STAR WEIGHTS LOOP (Air Distance Heuristic) : \n")
    run_astar_for_weights_in_range(heuristic_type = AirDistHeuristic, 
                                    problem=toy_map_problem,
                                    n = 30,
                                    low_heuristic_weight=0.5,
                                    high_heuristic_weight=0.95)


# --------------------------------------------------------------------
# ---------------------------- MDA Problem ---------------------------
# --------------------------------------------------------------------

loaded_problem_inputs_by_size = {}
loaded_problems_by_size_and_opt_obj = {}


def get_mda_problem(
        problem_input_size: str = 'small',
        optimization_objective: MDAOptimizationObjective = MDAOptimizationObjective.Distance):


    if (problem_input_size, optimization_objective) in loaded_problems_by_size_and_opt_obj:
        return loaded_problems_by_size_and_opt_obj[(problem_input_size, optimization_objective)]


    assert problem_input_size in {'small', 'moderate', 'big'}

    if problem_input_size not in loaded_problem_inputs_by_size:
        loaded_problem_inputs_by_size[problem_input_size] = MDAProblemInput.load_from_file(
            f'{problem_input_size}_mda.in', streets_map)


    problem = MDAProblem(
        problem_input=loaded_problem_inputs_by_size[problem_input_size],
        streets_map=streets_map,
        optimization_objective=optimization_objective)

    loaded_problems_by_size_and_opt_obj[(problem_input_size, optimization_objective)] = problem
    return problem


def basic_mda_problem_experiments():
    print()
    print('Solve the MDA problem (small input, only distance objective, UniformCost).')

    small_mda_problem_with_distance_cost = get_mda_problem('small', MDAOptimizationObjective.Distance)

    # Ex.14
    # create an instance of `UniformCost`, solve the `small_mda_problem_with_distance_cost`
    #       with it and print the results.

    # print("\n\nMDA PROBLEM (UNIFORM COST) : \n")
    uc = UniformCost()
    res = uc.solve_problem(small_mda_problem_with_distance_cost)
    print(res)


def mda_problem_with_astar_experiments():
    print()
    print('Solve the MDA problem (moderate input, only distance objective, A*, MaxAirDist & SumAirDist & MSTAirDist heuristics).')

    moderate_mda_problem_with_distance_cost = get_mda_problem('moderate', MDAOptimizationObjective.Distance)

    # Ex.17
    # create an instance of `AStar` with the `MDAMaxAirDistHeuristic`,
    #       solve the `moderate_mda_problem_with_distance_cost` with it and print the results.

    # print("\n\nA STAR MEDIUM MDA (MDAMaxAirDistHeuristic Heuristic) : \n")
    a_star_1 = AStar(heuristic_function_type=MDAMaxAirDistHeuristic)
    res = a_star_1.solve_problem(problem=moderate_mda_problem_with_distance_cost)
    print(res)

    # Ex.20
    # create an instance of `AStar` with the `MDASumAirDistHeuristic`,
    #       solve the `moderate_mda_problem_with_distance_cost` with it and print the results.

    # print("\n\nA STAR MEDIUM MDA (MDASumAirDistHeuristic Heuristic) : \n")
    a_star_2 = AStar(heuristic_function_type=MDASumAirDistHeuristic)
    res = a_star_2.solve_problem(problem=moderate_mda_problem_with_distance_cost)
    print(res)


    # Ex.23
    #  create an instance of `AStar` with the `MDAMSTAirDistHeuristic`,
    #       solve the `moderate_mda_problem_with_distance_cost` with it and print the results.

    # print("\n\nA STAR MEDIUM MDA (MDAMSTAirDistHeuristic Heuristic) : \n")
    a_star_3 = AStar(heuristic_function_type=MDAMSTAirDistHeuristic)
    res = a_star_3.solve_problem(problem=moderate_mda_problem_with_distance_cost)
    print(res)
    


def mda_problem_with_weighted_astar_experiments():
    print()
    print('Solve the MDA problem (small & moderate input, only distance objective, wA*).')

    small_mda_problem_with_distance_cost = get_mda_problem('small', MDAOptimizationObjective.Distance)
    moderate_mda_problem_with_distance_cost = get_mda_problem('moderate', MDAOptimizationObjective.Distance)

    # Ex.25
    # Call here the function `run_astar_for_weights_in_range()`
    #       with `MDAMSTAirDistHeuristic`
    #       over the `small_mda_problem_with_distance_cost`.

    # print("\n\nA STAR MDA WEIGHTS LOOP (MDAMSTAirDistHeuristic) : \n")
    run_astar_for_weights_in_range(heuristic_type = MDAMSTAirDistHeuristic, 
                                    problem=small_mda_problem_with_distance_cost,
                                    n = 30,
                                    low_heuristic_weight=0.5,
                                    high_heuristic_weight=0.95)

    # Ex.25
    # Call here the function `run_astar_for_weights_in_range()`
    #       with `MDASumAirDistHeuristic`
    #       over the `moderate_mda_problem_with_distance_cost`.

    # print("\n\nA STAR MDA WEIGHTS LOOP (MDASumAirDistHeuristic) : \n")
    run_astar_for_weights_in_range(heuristic_type = MDASumAirDistHeuristic, 
                                    problem=moderate_mda_problem_with_distance_cost,
                                    n = 30,
                                    low_heuristic_weight=0.5,
                                    high_heuristic_weight=0.95)

def multiple_objectives_mda_problem_experiments():
    print()
    print('Solve the MDA problem (moderate input, distance & tests-travel-distance objectives).')

    moderate_mda_problem_with_distance_cost = get_mda_problem('moderate', MDAOptimizationObjective.Distance)
    moderate_mda_problem_with_tests_travel_dist_cost = get_mda_problem('moderate', MDAOptimizationObjective.TestsTravelDistance)

    # Ex.31
    # create an instance of `AStar` with the `MDATestsTravelDistToNearestLabHeuristic`,
    #       solve the `moderate_mda_problem_with_tests_travel_dist_cost` with it and print the results.

    # print("\n\nA STAR  (MDATestsTravelDistToNearestLabHeuristic Heuristic) : \n")
    # a_star_4 = AStar(heuristic_function_type=MDATestsTravelDistToNearestLabHeuristic)
    # res = a_star_4.solve_problem(problem=moderate_mda_problem_with_tests_travel_dist_cost)
    # print(res)
    

    # Ex.34
    #  Implement the algorithm A_2 described in this exercise in the assignment instructions.
    #       Create an instance of `AStar` with the `MDAMSTAirDistHeuristic`.
    #       Solve the `moderate_mda_problem_with_distance_cost` with it and store the solution's (optimal)
    #         distance cost to the variable `optimal_distance_cost`.
    #       Calculate the value (1 + eps) * optimal_distance_cost in the variable `max_distance_cost` (for eps=0.6).
    #       Create another instance of `AStar` with the `MDATestsTravelDistToNearestLabHeuristic`, and specify the
    #          param `open_criterion` (to AStar c'tor) to be the criterion mentioned in the A_2 algorithm in the
    #          assignment instructions. Use a lambda function for that. This function should receive a `node` and
    #          has to return whether to add this just-created-node to the `open` queue. Remember that in python
    #          you can pass an argument to a function by its name `some_func(argument_name=some_value)`.
    #       Solve the `moderate_mda_problem_with_tests_travel_dist_cost` with it and print the results.

    # print("\n\nA2 algorithm : \n")
    # print("\n\nA STAR  (MDAMSTAirDistHeuristic Heuristic) : \n")
    a_star_4 = AStar(heuristic_function_type=MDAMSTAirDistHeuristic)
    res = a_star_4.solve_problem(problem=moderate_mda_problem_with_distance_cost)
    
    optimal_distance_cost = res.solution_g_cost

    eps = 0.6
    max_distance_cost = (1 + eps) * optimal_distance_cost

    # print(f"optimal_distance_cost = {optimal_distance_cost}, max_distance_cost = {max_distance_cost}")


    # print("\n\nA STAR  (MDATestsTravelDistToNearestLabHeuristic Heuristic) : \n")
    a_star_4 = AStar(heuristic_function_type=MDATestsTravelDistToNearestLabHeuristic, open_criterion=(lambda node : node.cost.distance_cost <= max_distance_cost))
    res = a_star_4.solve_problem(problem=moderate_mda_problem_with_tests_travel_dist_cost)

    print(res)


def mda_problem_with_astar_epsilon_experiments():
    print()
    print('Solve the MDA problem (small input, distance objective, using A*eps, use non-acceptable '
          'heuristic as focal heuristic).')

    small_mda_problem_with_distance_cost = get_mda_problem('small', MDAOptimizationObjective.Distance)

    # Firstly solve the problem with AStar & MST heuristic for having a reference for #devs.
    # astar = AStar(MDAMSTAirDistHeuristic)
    # res = astar.solve_problem(small_mda_problem_with_distance_cost)
    # print(res)

    def within_focal_h_sum_priority_function(node: SearchNode, problem: GraphProblem, solver: AStarEpsilon):
        if not hasattr(solver, '__focal_heuristic'):
            setattr(solver, '__focal_heuristic', MDASumAirDistHeuristic(problem=problem))
        focal_heuristic = getattr(solver, '__focal_heuristic')
        return focal_heuristic.estimate(node.state)

    # Ex.39
    # Try using A*eps to improve the speed (#dev) with a non-acceptable heuristic.
    #  Create an instance of `AStarEpsilon` with the `MDAMSTAirDistHeuristic`.
    #       Solve the `small_mda_problem_with_distance_cost` with it and print the results.
    #       Use focal_epsilon=0.03, and max_focal_size=40.
    #       Use within_focal_priority_function=within_focal_h_sum_priority_function. This function
    #        (defined just above) is internally using the `MDASumAirDistHeuristic`.
    

    astar_epsilon = AStarEpsilon(heuristic_function_type = MDAMSTAirDistHeuristic,
                                within_focal_priority_function = within_focal_h_sum_priority_function,
                                focal_epsilon = 0.03,
                                max_focal_size = 40)
    res = astar_epsilon.solve_problem(small_mda_problem_with_distance_cost)
    print(res)


def mda_problem_anytime_astar_experiments():
    print()
    print('Solve the MDA problem (moderate input, only distance objective, Anytime-A*, '
          'MSTAirDist heuristics).')

    moderate_mda_problem_with_distance_cost = get_mda_problem('moderate', MDAOptimizationObjective.Distance)

    # Ex.41
    #  create an instance of `AnytimeAStar` once with the `MDAMSTAirDistHeuristic`, with
    #       `max_nr_states_to_expand_per_iteration` set to 150, solve the
    #       `moderate_mda_problem_with_distance_cost` with it and print the results.

    anytime_astar = AnytimeAStar(heuristic_function_type = MDAMSTAirDistHeuristic,
                                max_nr_states_to_expand_per_iteration = 150,
                                initial_high_heuristic_weight_bound = 0.9)

    res = anytime_astar.solve_problem(moderate_mda_problem_with_distance_cost)
    print(res)


def run_all_experiments():
    print('Running all experiments')
    toy_map_problem_experiments()
    basic_mda_problem_experiments()

 
    mda_problem_with_astar_experiments()
    mda_problem_with_weighted_astar_experiments()
    multiple_objectives_mda_problem_experiments()
    mda_problem_with_astar_epsilon_experiments()
    mda_problem_anytime_astar_experiments()


if __name__ == '__main__':
    run_all_experiments()
