from __future__ import annotations

import math
import os
import random

import numpy as np
import pandas as pd

from configs import ProblemType, ObjectiveFunction, OUTPUT_PATH

random.seed(17)


def calculate_distances(locations):
    """Compute the distance matrix for the locations in dataset
    :param locations: list of location coordinates
    :return: distance matrix list
    """
    matrix = []
    for x in locations:
        curr_dist = []
        for y in locations:
            curr_dist.append(math.dist([x[0], x[1]], [y[0], y[1]]))
        matrix.append(curr_dist)

    return np.array(matrix)


def calculate_max_drone_distance(locations, distances):
    """Calculate maxium drone distance
    :param locations: List of location coordinates
    :param distances: Matrix of Euclidean distances
    :return max_distance: Maximum problem distance
    """
    n = len(locations);  # Number of nodes
    max_distance = -1
    for a in range(n):
        for b in range(n):
            for c in range(n):
                if a != b and a != c and b != c:
                    # Calculate the drone distance for the current combination
                    current_distance = distances[a, b] + distances[b, c]

                    # Update the maximum distance if necessary
                    max_distance = max(max_distance, current_distance)

    return max_distance.round(4)


class TSP:

    def __init__(self,
                 filename: str,
                 population_size: int = 150,
                 elite_population_size: int = 22,
                 mutation_rate: float = 0.30,
                 no_iterations: int = 1000,
                 truck_speed: float = 10,
                 drone_speed: float = 20,
                 truck_delivery_time: float = 0.1,
                 drone_delivery_time: float = 0.1,
                 problem_type: ProblemType = ProblemType.VRPD_CLUSTER,
                 stop_cond: int = 360,
                 objective_function: ObjectiveFunction = ObjectiveFunction.DELIVERY_TIME_MAX,
                 truck_capacity=0,
                 verbose: bool = False):
        """ Initialize Files class
        :param filename: Instance file name to be load
        :param population_size: Size of the population
        :param elite_population_size: Size of the elite population list
        :param mutation_rate: Changes of a gene mutation occuring
        :param no_iterations: Max. number of iterations/generations for the problem
        :param truck_speed: The constant speed of the truck
        :param drone_speed: The constant speed of the drone
        :param truck_delivery_time: Time the truck takes to deliver the load
        :param drone_delivery_time: Time the drone takes to deliver the load
        :param problem_type: vrpdi or vrpdi
        :param stop_cond: Stop conditions
        :param objective_function: Objective function
        :param truck_capacity: The capacity of the truck
        :param verbose: Whether to print logs during the execution of the class
        """
        # Create output directory
        if not os.path.isdir("data/output"):
            os.makedirs("data/output")

        # Create output/problem folder
        o_folder = filename.replace(".txt", "").strip().lower()
        if not os.path.isdir(f"data/output/{o_folder}"):
            os.makedirs(f"data/output/{o_folder}")

        self._verbose = verbose
        self._filename = filename
        self._truck_capacity, self._truck_speed, self._drone_speed, self._no_nodes, self._locations = self.load_instance \
            (self._filename)
        self._distances = calculate_distances(self._locations)
        self._population_size = population_size
        self._no_iterations = no_iterations
        self._truck_speed = truck_speed
        self._drone_speed = drone_speed
        self._truck_delivery_time = truck_delivery_time
        self._drone_delivery_time = drone_delivery_time
        self._problem_type = problem_type
        self._stop_cond = stop_cond
        self._max_drone_distance = calculate_max_drone_distance(self._locations, self._distances)
        self._objective_function = objective_function

        if truck_capacity > 0:
            self._truck_capacity = truck_capacity

        if mutation_rate > 0:
            self._mutation_rate = mutation_rate
        else:
            self._mutation_rate = float(random.uniform(0.15, 0.25))  # Mutation chances between 15% and 25%

        if elite_population_size > 0:
            self._elite_population_size = elite_population_size
        else:
            self._elite_population_size = int(population_size * 0.1) + 1  # Set to 10% of the population size

        self._no_trucks = self.calculate_no_trucks(self._no_nodes, self._truck_capacity)

    @property
    def distances(self):
        return self._distances

    @property
    def drone_delivery_time(self):
        return self._drone_delivery_time

    @property
    def drone_speed(self):
        return self._drone_speed

    @property
    def elite_population_size(self):
        return self._elite_population_size

    @property
    def filename(self):
        return self._filename

    @property
    def locations(self):
        return self._locations

    @property
    def max_drone_distance(self):
        return self._max_drone_distance * .75

    @property
    def mutation_rate(self):
        return self._mutation_rate

    @property
    def no_iterations(self):
        return self._no_iterations

    @property
    def no_nodes(self):
        return self._no_nodes

    @property
    def no_trucks(self):
        return self._no_trucks

    @property
    def objective_function(self):
        return self._objective_function

    @property
    def population_size(self):
        return self._population_size

    @property
    def problem_type(self):
        return self._problem_type

    @property
    def truck_capacity(self):
        return self._truck_capacity

    @property
    def truck_delivery_time(self):
        return self._truck_delivery_time

    @property
    def truck_speed(self):
        return self._truck_speed

    @property
    def verbose(self):
        return self._verbose

    @property
    def stop_cond(self):
        return self._stop_cond

    def open_file(self, filename, mode):
        """Open file
        :param filename: Filename to be opened
        :param mode: Open mode
        :return file: The opened file
        """
        try:
            f = open(filename, mode)

        except Exception as e:
            if self.verbose:
                print(f"Exception: {e}")
            raise Exception(f"Unable to open file: {filename}, error: {e}")

        return f

    def load_instance(self, filename: str) -> tuple:
        """ Load TSPD instances from Google Drive
        :param filename: File to be loaded
        :return tsp_instance_data: TSPD instance data
        """
        assert filename != "", "Filename cannot be empty"

        f = self.open_file(f"data/datasets/{filename}.txt", "r")

        truck_speed, drone_speed, nodes, depot, locations = 0, 0, 0, (0.0, 0.0), []
        for counter, x in enumerate(f):
            if counter in [0, 2, 4, 6, 8]:
                continue

            if counter == 1:
                truck_speed = self.to_float(x.strip())

            elif counter == 3:
                drone_speed = self.to_float(x.strip())

            elif counter == 5:
                nodes = int(x.strip())

            elif counter == 7:
                coord = x.strip().split(' ')
                depot = (self.to_float(coord[0]), self.to_float(coord[1]))

            else:
                coord = x.strip().split(' ')
                locations.append((self.to_float(coord[0]), self.to_float(coord[1])))

        f.close()
        locations.insert(0, depot)  # Insert depot coordinates at index 0 of locations list
        no_nodes = nodes - 1  # Deduct 1 from no_nodes to account for depot location at index 0 of location list
        truck_capacity = 40 if no_nodes < 100 else 100

        return truck_capacity, truck_speed, drone_speed, no_nodes, locations

    def calculate_no_trucks(self, no_nodes: int, truck_capacity: int, demand_per_node: int = 1) -> int:
        """Calculate the number of trucks for the problem
        :param no_nodes: Number of nodes of the problem
        :param truck_capacity: Capacity per truck
        :param demand_per_node: The demand for each of the nodes
        :return no_trucks: The number of trucks required by the problem
        """
        no_trucks = math.ceil((demand_per_node * no_nodes) / truck_capacity)

        if self._verbose:
            print(f"No Nodes: {no_nodes}, No Trucks: {no_trucks}")

        return no_trucks

    def to_float(self, _str) -> float:
        """Convert string to float
        :param _str: String value
        :return f: Converted float value
        """
        f = 0.0
        try:
            f = float(_str)
        except Exception as _e:
            if self.verbose:
                print(f"Unable to convert string: {_str} to float! \nException: {_e}")
        return f

    def file_contains(self, filename, text):
        """Check if output file contains text
        :param filename: Name of the file
        :param text: The text being checked
        :return contains: True if the file contains the text, False if otherwise
        """
        f = self.open_file(filename, "r")
        lines = f.readlines()
        f.close()

        contains = False
        for line in lines:
            if str(text).lower() in line.lower():
                contains = True

        return contains

    def write_to_file(self, filename, line=None, delete_if_exists: bool = False):
        """Write content to file
        :param filename: Name of the file
        :param delete_if_exists: Delete the file if already exists
        :param line: Line to be written
        """
        if delete_if_exists and os.path.exists(filename):
            try:
                os.remove(filename)
            except Exception as _e:
                raise Exception(f"Unable to delete file: {filename}")

        f = self.open_file(filename, "a")

        if line is not None:
            line = line.strip()  # Remove whitespaces
            if line[-2] != "\n":  # Check if line ends in newline character, if not add it
                line += "\n"
            f.write(line)
        f.close()

    def count_lines(self, filename):
        """Count the number of lines in a file
        :param filename: The fil name to be read
        :return lines: Lines in the file
        """
        f = self.open_file(filename, "r")
        lines = len(f.readlines())
        f.close()

        return lines

    def output_filename(self, file_type: str = "summary", experiment_no: int = 0, ext_type: str = "txt"):
        """ Creates output filenames
        :param ext_type: file extension
        :param file_type: Type of file
        :param experiment_no: Number of experiment
        :return file_name: File name
        """
        o_filename = OUTPUT_PATH.replace("{folder}", self.filename.replace(".txt", ""))
        o_filename = o_filename.replace("{filename}", file_type.strip().lower())

        if experiment_no > 0:
            o_filename += f"_{str(experiment_no)}"

        # {file_name}_{problem_type}_{obj_fn}_{max_iter}_{population}_{elitism}_{mutation}.{ext_type}
        o_filename += f"_{self.problem_type.value.lower()}_{self.objective_function.value.lower()}"
        o_filename += f"_{int(self.no_iterations)}_{int(self.population_size)}_{self.elite_population_size}"
        o_filename += f"_{int(self.mutation_rate * 100)}.{ext_type}"

        return o_filename

    def get_x_y_locations(self):
        locations = self.locations[1:len(self.locations)]
        locations = list(zip(*locations))

        df = pd.DataFrame(columns=['X', 'Y'])
        df['X'] = list(locations[0])
        df['Y'] = list(locations[1])

        return df.iloc[:, 0:3]
