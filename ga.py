from __future__ import annotations

import operator

from matplotlib import pyplot as plt

from configs import ObjectiveFunction, PlottingObject
from tsp import TSP
from solution import Solution
import random
import numpy as np
import pandas as pd
from collections import OrderedDict

random.seed(17)
np.random.seed(17)

color_arr = np.array(
    [(168, 50, 50), (168, 50, 168), (50, 160, 168), (156, 168, 50), (50, 113, 168),
     (168, 50, 101), (50, 168, 123), (168, 121, 50), (107, 168, 50), (97, 50, 168)]
) / 255


# Genetic Algorithm Class
def mating_pool(population: list, selection_results: list) -> list:
    """Select a mating pool from the population
    :param population: List
    :param selection_results: indexes what were selected for mating
    :return pool: Pool
    """
    pool = []
    for i in range(0, len(selection_results)):
        index = selection_results[i]
        pool.append(population[index])
    return pool


def sort_legend(keys: list, values: list) -> tuple[list, list]:
    """Order graph legends
    :param keys: Legend keys
    :param values: Legend values
    :return sorted_keys, sorted_values: Sorted keys and values
    """
    _dict = OrderedDict(zip(values, keys))
    keys_list = list(_dict.keys())
    keys_list.sort()
    sorted_dict = {i: _dict[i] for i in keys_list}

    return list(sorted_dict.keys()), list(sorted_dict.values())


def generation_fitness_values(population):
    """Convert generation fitness array into string for logging
    :param population: Generation population list in ranking order
    :return str_fitness: Comma separated string of distances
    """
    str_fitness = ""
    for solution in population:
        str_fitness += f"{solution[1]:.3f},"

    str_fitness = str_fitness[:-1]  # Remove last comma

    return str_fitness


class GeneticAlgorithm:

    def __init__(self, tsp: TSP):
        """
        :param tsp: TSP Instance
        """
        assert tsp.no_trucks > 0, "Number of trucks needs to be greater than 0"
        assert tsp.no_nodes > 0, "Number of nodes needs to be greater than 0"
        assert len(tsp.locations) > 0, "Number of location nodes needs to be greater than 0"

        self._tsp = tsp
        self._best = None

    def set_best(self, best):
        self._best = best

    @property
    def tsp(self):
        return self._tsp

    @property
    def best(self):
        return self._best

    def initial_population(self) -> list:
        """Generate the initial list of population
        :return population: List of candidate solutions
        """
        if self.tsp.verbose:
            print("Initializing population")

        population = []
        for _ in range(self.tsp.population_size):
            solution = Solution(self.tsp)
            population.append(solution)

            if self.tsp.verbose:
                print(f"Solution: {solution}")

        return population

    def rank_population(self, population: list) -> list:
        """Return a sorted list of the population size
        :return: Sorted ranked population list by distance
        """
        if self.tsp.verbose:
            print(f"Start population ranking by: {self.tsp.objective_function.value}")

        ranked_population = {}
        if self.tsp.objective_function == ObjectiveFunction.DISTANCE:
            for i in range(0, len(population)):
                population[i].calculate_totals()
                ranked_population[i] = population[i].total_distance
        else:
            for i in range(0, len(population)):
                population[i].calculate_totals()
                ranked_population[i] = population[i].total_delivery_time

        if self.tsp.verbose:
            print("Finished population ranking")

        return sorted(ranked_population.items(), key=operator.itemgetter(1), reverse=False)

    def selection(self, ranked_population: list) -> list:
        """Pick mating partners from the population list
        :param ranked_population: Ranked population list
        :return selection: Mating partners selection
        """
        selection = []
        df = pd.DataFrame(np.array(ranked_population), columns=["Index", "Fitness"])
        df['cum_sum'] = df.Fitness.cumsum()
        df['cum_perc'] = 100 * (df.cum_sum / df.Fitness.sum())

        for i in range(0, self.tsp.elite_population_size):
            selection.append(ranked_population[i][0])

        for i in range(0, len(ranked_population) - self.tsp.elite_population_size):
            pick = 100 * random.random()
            for i in range(0, len(ranked_population)):
                if pick <= df.iat[i, 3]:
                    # Select indexes from the ranked population
                    selection.append(ranked_population[i][0])
                    break

        return selection

    def crossover(self, parent1: Solution, parent2: Solution) -> Solution:
        """Crossover two candidate solutions
        :param parent1: Candidate solution to be used as parent 1
        :param parent2: Candidate solution to be used as parent 2
        :return: Resulting candidate solution
        """
        node_p1 = []
        delivery_p1 = []
        node_p2 = []
        delivery_p2 = []
        gene_a = int(random.random() * len(parent1.node_sequence))
        gene_b = int(random.random() * len(parent1.node_sequence))
        start, end = min(gene_a, gene_b), max(gene_a, gene_b)

        for i in range(start, end):
            node_p1.append(parent1.node_sequence[i])
            delivery_p1.append(parent1.deliveries[i])

        i = 0
        for item in parent2.node_sequence:
            if item not in node_p1:
                node_p2.append(item)
                delivery_p2.append(parent2.deliveries[i])

            i += 1

        node_sequence = node_p1 + node_p2
        deliveries = delivery_p1 + delivery_p2

        return Solution(self.tsp, node_sequence, deliveries)

    def crossover_population(self, mating_pool) -> list:
        """Crossover the population
        :param mating_pool: Mating pool to be used
        :return children: Resulting children from the pool
        """
        children = []
        length = (len(mating_pool) - self.tsp.elite_population_size)
        pool = random.sample(mating_pool, len(mating_pool))
        for i in range(0, self.tsp.elite_population_size):
            children.append(mating_pool[i])

        for i in range(0, length):
            child = self.crossover(pool[i], pool[len(mating_pool) - i - 1])
            children.append(child)

        return children

    def mutate(self, solution: Solution) -> Solution:
        """Perform mutation by swapping genes (nodes)
        :param solution: Solution to be mutated
        :return individual: Solution
        """
        for swap in range(len(solution.node_sequence)):
            if random.random() < self.tsp.mutation_rate:
                swap_with = swap
                while swap == swap_with:
                    # Prevent both swapping indexes being equal
                    swap_with = int(random.random() * len(solution.node_sequence))
                gene_a, gene_b = solution.node_sequence[swap], solution.node_sequence[swap_with]
                solution.set_node(gene_a, swap)
                solution.set_node(gene_b, swap_with)

        if not solution.tsp.problem_type.is_vrp():
            for i in range(len(solution.deliveries)):
                if random.random() < solution.tsp.mutation_rate:
                    # bit flip
                    gene = 0 if solution.deliveries[i] == 1 else 1
                    solution.set_delivery(gene, i)
        return solution

    def mutate_population(self, population) -> list:
        """Perform population mutation
        :param population: The population to be mutated
        :return populated: Population
        """
        mutated_population = []
        for index in range(0, len(population)):
            mutated_individual = self.mutate(population[index])
            mutated_population.append(mutated_individual)

        return mutated_population

    def next_generation(self, cur_generation) -> list:
        """Create the next generation of the population
        :param cur_generation: The current generation of the population
        :return next_generation: The next generation of the population to be evaluated
        """
        next_generation = self.rank_population(cur_generation)
        next_generation = self.selection(next_generation)
        next_generation = mating_pool(cur_generation, next_generation)
        next_generation = self.crossover_population(next_generation)
        next_generation = self.mutate_population(next_generation)
        return next_generation

    def plot(self, fig_size=(8, 4), truck_drone_ids=None, experiment_no=1, fig_objects=None, save_file_only=True):
        """Plot the complete solution graph for the VRPDi
        :param fig_size: Size of the image to be generated
        :param truck_drone_ids: List of specific truck-drone system ids to be plotted
        :param experiment_no: Experiment number
        :param fig_objects: List of objects to be plotted in the graph
        :param save_file_only: If True, save the figures only
        """
        if fig_objects is None:
            fig_objects = [PlottingObject.ALL]

        assert self.best is not None, "Please run the Genetic Algorithm before plotting routes"

        candidate_solution = self.best

        if not fig_objects or len(fig_objects) == 0:
            return None

        plt.figure(figsize=fig_size)
        plt.rcParams["figure.autolayout"] = True

        locations = self.tsp.locations
        if (truck_drone_ids is None) or (len(truck_drone_ids) == 0):
            trucks = range(self.tsp.no_trucks)
        else:
            trucks = truck_drone_ids

        number_of_trucks = len(trucks)
        plot_all = (PlottingObject.ALL in fig_objects)
        plot_trucks = plot_all or (PlottingObject.TRUCK in fig_objects)
        plot_drones = plot_all or (PlottingObject.DRONE in fig_objects)
        plot_customer_nodes = plot_all or (PlottingObject.CUSTOMER_NODE in fig_objects)
        plot_interceptions = plot_all or (PlottingObject.INTERCEPTION in fig_objects)

        x, y = [], [],
        for loc in locations:
            x.append(loc[0])
            y.append(loc[1])

        # Costumer nodes
        plt.plot(x, y, 'ko')

        if plot_customer_nodes:
            # Label customer nodes
            plt.text(locations[0][0] - 1, locations[0][1] - 5, "Depot")
            for i in range(1, len(locations) - 1):
                plt.text(locations[i][0] - 1, locations[i][1] + 2, f"Node {i}")

        display_interception_label = True
        if number_of_trucks > 0:

            if (candidate_solution.truck_routes or candidate_solution.drone_routes) \
                    and (len(candidate_solution.truck_routes) > 0 or len(candidate_solution.drone_routes) > 0):

                for truck in trucks:
                    truck_route = candidate_solution.truck_routes[truck]
                    interceptions = candidate_solution.interceptions[truck]
                    color = color_arr[truck]

                    # Truck route
                    if plot_trucks:
                        x_truck, y_truck = [], []
                        for i in truck_route:
                            x_truck.append(locations[i][0])
                            y_truck.append(locations[i][1])
                        plt.plot(x_truck, y_truck, linestyle="-", color=color, label="Truck " + str(truck + 1))

                    # Drone route
                    if plot_drones:
                        display_drone_label = True
                        drone_flying_coordinates = candidate_solution.build_drone_flying_route(truck)
                        for coordinates in drone_flying_coordinates:
                            x_drone, y_drone = [], []
                            for coordinate in coordinates:
                                x_drone.append(coordinate[0])
                                y_drone.append(coordinate[1])
                            if len(x_drone) > 0 and len(y_drone) > 0:
                                a_label = "Drone " + str(truck + 1) if display_drone_label else ""
                                plt.plot(x_drone, y_drone, linestyle="--", color=color, label=a_label)
                                display_drone_label = False

                    # Interception
                    if plot_interceptions and len(interceptions) > 0:
                        x_int, y_int = [], []
                        for interception in interceptions:
                            if len(interception) > 0 \
                                    and all(inter > 0 for inter in interception) \
                                    and (float(interception[0]) <= max(x) and float(interception[1]) <= max(y)):
                                x_int.append(interception[0])
                                y_int.append(interception[1])

                        plt.plot(x_int, y_int, 'ks', label="Interception" if display_interception_label else "")
                        display_interception_label = False

            xlim, ylim = -10, -10
            if min(x) < 0 and min(y) < 0:
                xlim, ylim = min(x) * 1.25, min(y) * 1.25

            plt.xlim(xlim, max(x) * 1.25)
            plt.ylim(ylim, max(y) * 1.25)

            if (PlottingObject.ALL in fig_objects) or (PlottingObject.LEGEND in fig_objects):
                legend_keys, legend_values = plt.gca().get_legend_handles_labels()
                sorted_legend_keys, sorted_legend_values = sort_legend(legend_keys, legend_values)
                plt.legend(sorted_legend_values, sorted_legend_keys)

            if save_file_only:
                filename = self.tsp.output_filename("deliveries", experiment_no, "png")
                plt.savefig(filename)
                plt.close()

            else:
                plt.show()

    def to_nodes(self, flying_coordinates):
        """Convert drone flying route coordinate to node list
        :param flying_coordinates: Coordinates to be converted
        :return nodes: List of nodes
        """
        nodes = []
        for flying_coordinate in flying_coordinates:
            route = []
            for coordinate in flying_coordinate:
                for node, location in enumerate(self.tsp.locations):
                    if coordinate[0] == location[0] and coordinate[1] == location[1]:
                        route.append(node)
                        break
                    if node == len(self.tsp.locations) - 1:
                        route.append(coordinate)
                        break
                nodes.append(route)

        return nodes

    def log(self, message) -> None:
        """Log message if verbose allowed
        :param message: Message to be logged
        """
        if self.tsp.verbose:
            print(message)

    def run(self, save: bool = False, experiment_no=0) -> None:
        """Execute evolution
        :param save: Whether to save progress logs to a file
        :param experiment_no: Experiment number
        """
        last_improved = 0
        if self.tsp.verbose:
            print("Starting ...")

        population = self.initial_population()
        ranked_population = self.rank_population(population)
        self.set_best(population[ranked_population[0][0]])

        if self.tsp.verbose:
            print(f"Best: {self.best}")

        d_filename, o_filename, lines = None, None, 0
        if save:
            # Costs
            o_filename = self.tsp.output_filename("costs", experiment_no)
            self.tsp.write_to_file(o_filename, delete_if_exists=True)
            lines = self.tsp.count_lines(o_filename)

            # Diversity
            d_filename = self.tsp.output_filename("diversity", experiment_no)
            self.tsp.write_to_file(d_filename, delete_if_exists=True)

        lines += 1
        stopped_iter = self.tsp.no_iterations
        sort_by_distance = ObjectiveFunction.DISTANCE == self.tsp.objective_function
        for iter_no in range(1, self.tsp.no_iterations + 1):

            if (self.tsp.stop_cond > 0) and (last_improved >= (self.tsp.stop_cond + 1)):
                _stopped_iter = iter_no
                break

            if self.tsp.verbose:
                print(f"Generation: {iter_no}")

            population = self.next_generation(population)
            ranked_population = self.rank_population(population)
            candidate_solution = population[ranked_population[0][0]]

            update_best = (sort_by_distance and (candidate_solution.total_distance < self.best.total_distance)) or \
                          (not sort_by_distance and (
                                  candidate_solution.total_delivery_time < self.best.total_delivery_time))

            if update_best:
                last_improved = 0  # Reset counter
                self.set_best(candidate_solution)

                if self.tsp.verbose:
                    print(f"Found better solution: {candidate_solution}")

            if save:
                # Costs
                cost_msg = f"{lines}, {candidate_solution.total_distance:.3f}, {candidate_solution.total_delivery_time:.3f}"
                self.tsp.write_to_file(o_filename, cost_msg)

                # Diversity
                self.tsp.write_to_file(d_filename, generation_fitness_values(ranked_population))
            lines += 1
            last_improved += 1

        self.best.set_generations(stopped_iter)

        if self.tsp.verbose:
            print("Ending execution ...")
