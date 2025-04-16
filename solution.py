from __future__ import annotations

import copy
import math
import random

import numpy as np
import scipy as sp
from sklearn.cluster import KMeans

from configs import Feasibility, ObjectiveFunction
from tsp import TSP

random.seed(17)
np.random.seed(17)


def delivery_mode(delivered_by_drone: int) -> str:
    """Get delivery mode
    :param delivered_by_drone: Delivery mode
    :return delivery_mode: Mode of delivery
    """
    return "Truck" if delivered_by_drone == 0 else "Drone"


def angle_distance(position_1: tuple, position_2: tuple) -> float:
    """Calculate the angle of the position for interception
    :param position_2: Position 1 coordinates
    :param position_1: Position 2 coordinates
    :return angle: Angle
    """
    x1, y1, x2, y2 = position_1[0], position_1[1], position_2[0], position_2[1]

    if x2 - x1 == 0 and y2 - y1 > 0:
        a = 90
    elif x2 - x1 == 0 and y2 - y1 < 0:
        a = 270
    elif y2 - y1 >= 0 and x2 - x1 > 0:
        a = math.degrees(math.atan((y2 - y1) / (x2 - x1)))

    elif (y2 - y1 >= 0 > x2 - x1) or (y2 - y1 < 0 and x2 - x1 < 0):
        a = math.degrees(math.atan((y2 - y1) / (x2 - x1))) + 180
    elif y2 - y1 < 0 < x2 - x1:
        a = math.degrees(math.atan((y2 - y1) / (x2 - x1))) + 360
    else:
        a = 0
    return a


def calculate_distance(coordinate_one, coordinate_two) -> float:
    """Calculate the Euclidean distance between two coordinates
    :param coordinate_one: Coordinate 1 in format (X, Y)
    :param coordinate_two: Coordinate 2 in format (X, Y)
    :return distance: The Euclidean distance
    """
    return math.dist([coordinate_one[0], coordinate_two[0]], [coordinate_one[1], coordinate_two[1]])


# Solution
class Solution:

    def __init__(self, tsp: TSP, node_sequence: list = None, deliveries: list = None, load_assignment=None):
        """
        :param node_sequence: Visited customer nodes sequence
        :param deliveries: Delivery vehicle sequence
        :param load_assignment: Load assignment distribution
        """
        if tsp.verbose:
            print("Creating solution")
        assert tsp.no_trucks > 0, "Number of trucks needs to be greater than 0"
        assert tsp.no_nodes > 0, "Number of nodes needs to be greater than 0"
        assert len(tsp.locations) > 0, "Number of location nodes needs to be greater than 0"

        node_set = (node_sequence and deliveries) or (not node_sequence and not deliveries)

        assertion_msg = "The two variables 'node_sequence' and 'deliveries' need to be initialized"
        assert node_set, assertion_msg

        if load_assignment is None:
            load_assignment = []
        if deliveries is None:
            deliveries = []
        if node_sequence is None:
            node_sequence = []

        self._tsp = tsp
        self._node_sequence = node_sequence
        self._deliveries = deliveries
        self._load_assignment = load_assignment
        self._total_distance = 0
        self._total_delivery_time = 0
        self._total_waiting_time = 0
        self._interceptions = []
        self._truck_routes = []
        self._drone_routes = []
        self._drone_positions = []
        self._tour_delivery_distances = []
        self._tour_delivery_times = []
        self._tour_delivery_waiting_times = []
        self._generations = 0
        self._trucks_cluster = []

        if self._tsp.problem_type.is_vrp():
            self._deliveries = [0] * self._tsp.no_nodes

        else:
            if not self._deliveries:
                if self._tsp.no_nodes < 49:
                    self._deliveries = []
                    [self._deliveries.append(random.randint(0, 1)) for _ in range(self.tsp.no_nodes)]

                else:
                    self._deliveries = [int(0)] * self._tsp.no_nodes
                    for i in range(len(self._deliveries) - 1):
                        if (random.random() < self._tsp.mutation_rate) and (self._deliveries[i + 1] == 0):
                            self._deliveries[i] = int(1)

        if not self._load_assignment:
            self._load_assignment, self._trucks_cluster = self.assign_truck_load()

        if not self._node_sequence:

            if self.tsp.problem_type.is_cluster():
                # Define node sequence based on clustering
                self._node_sequence = []
                for truck_id in range(self._tsp.no_trucks):
                    ix = np.array([i for i, x in enumerate(self._trucks_cluster) if x == truck_id]) + 1
                    self._node_sequence += sorted(ix, key=lambda x: random.random())
            else:
                self._node_sequence = random.sample(range(1, self._tsp.no_nodes + 1), self._tsp.no_nodes)

        self._chromosome = copy.deepcopy(self._node_sequence)

        for i in self._deliveries:
            self._chromosome.append(i)

        for i in self._load_assignment:
            self._chromosome.append(i)

        if self._tsp.verbose:
            prt_msg = f"Solution::[Nodes: {self._node_sequence},"
            prt_msg += f"Deliveries: {self._deliveries}, Assignment: {self._load_assignment}]\n"
            print(prt_msg)

    @property
    def chromosome(self):
        self._chromosome = copy.deepcopy(self._node_sequence)
        for i in self._deliveries:
            self._chromosome.append(i)
        for i in self._load_assignment:
            self._chromosome.append(i)
        return self._chromosome

    @property
    def deliveries(self):
        return self._deliveries

    @deliveries.setter
    def deliveries(self, deliveries):
        self._deliveries = deliveries

    @property
    def drone_positions(self) -> list:
        return self._drone_positions

    @property
    def drone_routes(self) -> list:
        return self._drone_routes

    @property
    def interceptions(self):
        return self._interceptions

    @property
    def generations(self):
        return self._generations

    def set_generations(self, generations):
        self._generations = generations

    @property
    def load_assignment(self):
        return self._load_assignment

    @property
    def node_sequence(self):
        return self._node_sequence

    def set_delivery(self, delivery, index):
        self._deliveries[index] = delivery

    def set_node(self, node, index):
        self._node_sequence[index] = node

    @property
    def total_delivery_time(self):
        if not self._total_delivery_time or self._total_delivery_time <= 0:
            self._total_distance, self._total_delivery_time, self._total_waiting_time = self.calculate_totals()
        return self._total_delivery_time

    @property
    def total_distance(self):
        if not self._total_distance or self._total_distance <= 0:
            self._total_distance, self._total_delivery_time, self._total_waiting_time = self.calculate_totals()
        return self._total_distance

    @property
    def total_waiting_time(self):
        if not self._total_waiting_time or self._total_waiting_time <= 0:
            self._total_distance, self._total_delivery_time, self._total_waiting_time = self.calculate_totals()
        return self._total_waiting_time

    @property
    def tour_delivery_distances(self):
        return self._tour_delivery_distances

    @property
    def tour_delivery_times(self):
        return self._tour_delivery_times

    @property
    def tour_delivery_waiting_times(self):
        return self._tour_delivery_waiting_times

    @property
    def tsp(self):
        return self._tsp

    @property
    def trucks_cluster(self) -> list:
        return self._trucks_cluster

    @property
    def truck_routes(self) -> list:
        return self._truck_routes

    def __repr__(self):
        """Represet Individual object as String"""

        return f"({self.chromosome}, {self._total_distance: .4f})"

    def assign_truck_load(self) -> tuple:
        """Return a randomly chosen list of n non-negative integers summing to total.
            Each such list is equally likely to occur.
        :return load_assignment: Truck load assignment distribution
        """
        y_pred = []
        load_assignments = [0] * self.tsp.no_trucks
        if self.tsp.problem_type.is_cluster():
            X = self.tsp.get_x_y_locations()
            kmeans = KMeans(n_clusters=int(self.tsp.no_trucks), random_state=17, n_init='auto').fit(X)
            y_pred = list(kmeans.predict(X))

            for i in y_pred:
                load_assignments[i] += 1

        else:
            # Specific for 500 nodes, and 5 trucks
            if self.tsp.no_nodes == 499 and self.tsp.no_trucks == 5:
                return [100, 100, 100, 100, 99], y_pred

            load_assignments = [0] * self.tsp.no_trucks
            while (
                    (0 in load_assignments) or
                    any(load > self.tsp.truck_capacity for load in load_assignments)
            ):
                truck_load = sorted(random.choices(range(0, self.tsp.no_nodes), k=self.tsp.no_trucks - 1))
                load_assignments = [x - y for x, y in zip(truck_load + [self.tsp.no_nodes], [0] + truck_load)]

        return load_assignments, y_pred

    def calculate_totals(self) -> tuple:
        """Calculate the total distance and waiting time taken to complete the deliveries
        :return total_distance, total_waiting_time: Total distance, Toal waiting time
        """
        feasibility = self.is_feasible()

        if feasibility != Feasibility.FEASIBLE:
            self.log(f"Infeasible solution: {self} | Type: {feasibility.value}")
            self.repair(feasibility)
            self.log(f"Feasible solution: {self}")

        # Initialize lists
        self._interceptions = []
        self._drone_positions = []
        self._drone_routes = []
        self._truck_routes = []
        self._tour_delivery_distances = []
        self._tour_delivery_times = []
        self._tour_delivery_waiting_times = []

        start = 0
        end = 0
        delivery_mode_dict = {0: 'Truck', 1: 'Drone'}

        # Loop first for each of the truck-drone system
        for truck in range(self.tsp.no_trucks):
            no_of_deliveries = self.load_assignment[truck]
            end = start + no_of_deliveries
            deliveries = self.deliveries[start:end]
            nodes = self.node_sequence[start:end]

            # Determine launch drone positions
            drone_launch_index = []
            truck_route = [0]
            drone_route = []
            for i in range(len(deliveries)):

                if delivery_mode_dict[deliveries[i]] == 'Truck':  # Truck
                    truck_route.append(nodes[i])

                else:  # Drone
                    drone_route.append(nodes[i])  # Add to route list
                    if i >= 1:  # Add launch position and not landing position
                        launch_index = i - 1
                    else:
                        launch_index = 0
                    drone_launch_index.append(launch_index)

            start = end
            truck_route.append(0)  # Insert depot locations at the begining and the end of the truck_route

            # Calculate interceptions
            interceptions, tour_waiting_time = self.calculate_interception(
                truck_route, drone_route, drone_launch_index
            )

            # Calculate truck-drone system tour time
            tour_delivery_distance, tour_delivery_time = self.calculate_tour_values(
                truck_route, drone_route, tour_waiting_time
            )

            self.interceptions.append(interceptions)
            self.truck_routes.append(truck_route)
            self.drone_routes.append(drone_route)
            self.drone_positions.append(drone_launch_index)
            self.tour_delivery_distances.append(tour_delivery_distance)
            self.tour_delivery_times.append(tour_delivery_time)
            self.tour_delivery_waiting_times.append(tour_waiting_time)

        distance = sum(self.tour_delivery_distances)
        waiting_time = sum(self.tour_delivery_waiting_times)

        if ObjectiveFunction.DELIVERY_TIME_MEAN == self.tsp.objective_function:
            delivery_time = np.average(self.tour_delivery_times)
        else:
            delivery_time = max(self.tour_delivery_times)

        return distance, delivery_time, waiting_time

    def to_float(self, _str) -> float:
        """Convert string to float
        :param _str: value
        :return f: Converted float value
        """
        f = 0.0
        try:
            f = float(_str)
        except Exception as e:
            self.log(f"Unable to convert string: {_str}\nExcption: {e}")
        return f

    def calculate_tour_values(self, truck_route: list, drone_positions: list, waiting_time) -> tuple[float, float]:
        """Calculate the distance and time for each truck-drone system takes to complete its tour
        :param truck_route: List truck route nodes to be visited by the truck
        :param drone_positions: Drone launch positions in relation to the truck route nodes
        :param waiting_time: Waiting time
        :return total_distance: Total distance taken by the truck-drone system to visit all nodes assigned to it
        :return total_delivery_time: Total time taken by the truck-drone system to delivery into all nodes assigned to it
        """
        total_distance = 0
        total_delivery_time = waiting_time
        truck_time_to_deliver = self.tsp.distances / self.tsp.truck_speed

        for index in range(1, len(truck_route)):
            prev_node, curr_node = truck_route[index - 1], truck_route[index]

            if index not in drone_positions:
                total_distance += self.tsp.distances[prev_node, curr_node]
                total_delivery_time += truck_time_to_deliver[prev_node, curr_node]

        return total_distance, total_delivery_time

    def truck_drone_speed_difference(self) -> float:
        """ Calculate the speed difference between drone and truck
        :return speed_difference: Speed difference
        """
        return (self.tsp.drone_speed ** 2) - (self.tsp.truck_speed ** 2)

    def calculate_interception(self, truck_routes: list, drone_routes: list, drone_launch_index: list) -> tuple[
        list, float]:
        """Calculate possible interception points for the Truck-Drone system
        :param truck_routes: List of nodes to be fulfilled by the truck
        :param drone_routes: List of nodes to be fulfilled by the drone
        :param drone_launch_index: List indexes of the nodes the drone is launched from
        :return interception_points: Interception points
        :return wait: Truck waiting time
        """
        time_to_deliver = self.tsp.distances / self.tsp.truck_speed
        truck_route_length = len(truck_routes)
        drone_route_length = len(drone_routes)
        interception_points = [(0., 0.)] * drone_route_length
        interceptions = [0.] * truck_route_length
        interceptions_plus = [0] * truck_route_length
        interceptions_minus = [0] * truck_route_length
        truck_time = 0
        wait = 0
        h = 0
        z = 0
        locations = self.tsp.locations

        for i in range(truck_route_length - 1):
            temp_truck_time = truck_time
            delivery_time_to_next_node = time_to_deliver + truck_time
            truck_time = delivery_time_to_next_node[truck_routes[i]][truck_routes[i + 1]]
            truck_time += self.tsp.truck_delivery_time

            message = f"i: {i}\ndrone_route_length > 0: {drone_route_length > 0}\n"

            if drone_route_length > 0:
                message += f"{h} < len(drone_launch_index): {h < len(drone_launch_index)}\n"
                message += f"{z} < len(drone_route_length): {z < drone_route_length}\n"

                if h < len(drone_launch_index) and z < drone_route_length:
                    message += f"drone_launch_index[{h}]: {drone_launch_index[h]}\n"

                    if i == drone_launch_index[h]:
                        truck_i = temp_truck_time
                        truck_node = truck_routes[i]
                        next_truck_node = truck_routes[i + 1]
                        drone_node = drone_routes[z]

                        # Angle and distance from Node 1 to Node 2 (Truck)
                        angle = angle_distance(
                            (locations[truck_node][0], locations[truck_node][1]),
                            (locations[next_truck_node][0], locations[next_truck_node][1])
                        )

                        # Drone launch position
                        drone_launch_position = np.array([locations[truck_node][0], locations[truck_node][1]])

                        # Distance travelled by the drone
                        drone_distance = np.linalg.norm(
                            np.array([locations[truck_node][0], locations[truck_node][1]]) - np.array(
                                [locations[drone_node][0], locations[drone_node][1]]))

                        # Time taken by drone to fly from node i to j
                        time_taken = truck_i + drone_distance / self.tsp.drone_speed - truck_i

                        # Position of truck after drone has reached delivery point
                        truck_position_after_delivery = drone_launch_position + time_taken * self.tsp.truck_speed * np.array(
                            [np.cos(np.pi / 180 * angle), np.sin(np.pi / 180 * angle)])

                        dot1 = np.array([locations[drone_routes[z]][0], locations[drone_routes[z]]][
                                            1]) - truck_position_after_delivery
                        dot2 = np.array(
                            [np.cos(np.pi / 180 * angle), np.sin(np.pi / 180 * angle)]) * self.tsp.truck_speed
                        norm = np.array([truck_position_after_delivery[0], truck_position_after_delivery[1]] - np.array(
                            [locations[next_truck_node][0], locations[next_truck_node][1]]))
                        eq_1 = np.dot(dot1, dot2)
                        eq_2 = -2 * np.dot(dot1, dot2)
                        eq_3 = (2 * np.dot(dot1, dot2)) ** 2
                        eq_4 = 4 * self.truck_drone_speed_difference() * -1 * (np.linalg.norm(norm) ** 2)

                        # +/- Interceptions
                        interceptions_plus[z] = eq_2 + np.sqrt(eq_3 - eq_4)
                        interceptions_minus[z] = eq_2 - np.sqrt(eq_3 - eq_4)

                        message += f"h: {h}\nz: {z}\ntruck_node_location: {(locations[truck_node][0], locations[truck_node][1])}\n"
                        message += f"truck_node_next_location: {(locations[next_truck_node][0], locations[next_truck_node][1])}\n"
                        message += f"truck_node: {truck_node}\nnext_truck_node: {next_truck_node}\ndrone_node: {drone_node}\n"
                        message += f"angle: {angle}\ndrone_launch_position: {drone_launch_position}\ndrone_distance: {drone_distance}\n"
                        message += f"time_taken: {time_taken}\ntruck_position_after_delivery: {truck_position_after_delivery}\n"
                        message += f"truck_drone_speed_difference: {self.truck_drone_speed_difference()}\neq_1:{eq_1} \neq2: {eq_2}\neq3: {eq_3}\neq4: {eq_4}\n"
                        message += f"interceptions_plus[{z}]: {interceptions_plus[z]}\ninterceptions_minus[{z}]: {interceptions_minus[z]}\n"

                        if self.truck_drone_speed_difference() > 0:
                            eq_5 = interceptions_minus[z] / (2 * self.truck_drone_speed_difference())
                            eq_6 = interceptions_plus[z] / (2 * self.truck_drone_speed_difference())
                            interceptions[z] = eq_5 if all(v >= 0 for v in interceptions_minus) else eq_6
                            inter_point = truck_position_after_delivery + (self.tsp.truck_speed * np.array(
                                [np.cos(np.pi / 180 * angle), np.sin(np.pi / 180 * angle)])) * (
                                                  self.tsp.drone_delivery_time + interceptions[z])
                            interception_points[z] = (inter_point[0], inter_point[1])
                            message += f"eq_5: {eq_5}\neq_6: {eq_6}\ninterceptions[{z}]: {interceptions[z]}\n"

                        else:
                            eq_7 = sp.spatial.distance.pdist(np.array(
                                [[locations[drone_node][0], locations[drone_node][1]],
                                 [locations[next_truck_node][0], locations[next_truck_node][1]]]),
                                'euclidean') / self.tsp.drone_speed
                            interceptions[z] = eq_7[0]
                            message += f"interceptions[{z}]:{interceptions[z]}\n"

                        awaiting = truck_i + drone_distance / self.tsp.drone_speed + self.tsp.drone_delivery_time + \
                                   interceptions[z]
                        if awaiting >= truck_time:
                            interception_points[z] = (locations[next_truck_node][0], locations[next_truck_node][1])

                        message += f"truck_i: {truck_i}\ndrone_distance: {drone_distance}\ndrone_delivery_time: {self.tsp.drone_delivery_time}\ninterceptions[{z}]: {interceptions[z]}\n"
                        message += f"\nawaiting: {awaiting}\ntruck_time: {truck_time}\nawaiting >= truck_time: {awaiting >= truck_time}\ninterception_points[{z}]: {interception_points[z]}\n\n\n"
                        self.log(message)

                        awaiting = max(awaiting - truck_time, 0)
                        wait += awaiting
                        truck_time += awaiting
                        h += 1
                        z += 1

        return interception_points, wait

    def repair(self, repair_type: Feasibility) -> None:
        """Repair infeasible candidate solution to a feasible one
        :param repair_type: Type of reparation to be performed
        """
        repaired_deliveries, start, end = [], 0, 0
        if Feasibility.INFEASIBLE_REPEATED_DRONES == repair_type:
            for truck in range(self.tsp.no_trucks):
                no_of_deliveries = self.load_assignment[truck]
                end = start + no_of_deliveries
                deliveries = self.deliveries[start:end]

                if no_of_deliveries > 0:
                    for i in range(no_of_deliveries - 1):
                        if deliveries[i] == 1 and deliveries[i + 1] == 1:
                            deliveries[i] = 0
                        repaired_deliveries.append(deliveries[i])
                    repaired_deliveries.append(deliveries[no_of_deliveries - 1])
                start += no_of_deliveries
            self.deliveries = repaired_deliveries

        elif Feasibility.INFEASIBLE_DRONE_ENDURANCE == repair_type:
            # Set all deliveries to truck
            self.deliveries = [0] * self.tsp.no_nodes

    def is_feasible(self, max_repeated_drone_delivery_allowed: int = 2) -> Feasibility:
        """Determine whether the solution is feasible or not
        :param max_repeated_drone_delivery_allowed: Maximum number of repeats allowed
        :return: True if the solution is feasible, False if otherwise
        """
        start, end, feasiblility = 0, 0, Feasibility.FEASIBLE
        for truck in range(self.tsp.no_trucks):
            no_of_deliveries = self.load_assignment[truck]
            end = start + no_of_deliveries
            deliveries = self.deliveries[start:end]
            counter = 0
            for i in deliveries:
                counter = counter + 1 if i == 1 else 0
                if counter >= max_repeated_drone_delivery_allowed:
                    feasiblility = Feasibility.INFEASIBLE_REPEATED_DRONES
                    break
            start = end

        if (Feasibility.FEASIBLE == feasiblility) and self.tsp.problem_type.is_endurance():
            for truck in range(self.tsp.no_trucks):
                drone_flying_routes = self.build_drone_flying_route(truck)

                for drone_flying_route in drone_flying_routes:
                    distance = 0
                    for i in range(len(drone_flying_route) - 1):
                        distance += math.dist(
                            [drone_flying_route[i][0], drone_flying_route[i + 1][0]],
                            [drone_flying_route[i][1], drone_flying_route[i + 1][1]]
                        )

                    if distance >= self.tsp.max_drone_distance:
                        feasiblility = Feasibility.INFEASIBLE_DRONE_ENDURANCE
                        break

                if Feasibility.FEASIBLE != feasiblility:
                    break

        return feasiblility

    def build_drone_flying_route(self, truck_drone_pair_id: int) -> list:
        """Build drone complete flying route
        :param truck_drone_pair_id: ID of the truck-drone pair
        :return drone_flying_route: List of coordenates the drone will
        launch from and fly to as well as the interception coordernates
        """
        drone_flying_route = []
        start = 0 if truck_drone_pair_id == 0 else sum(self.load_assignment[:truck_drone_pair_id])
        end = start + self.load_assignment[truck_drone_pair_id]
        customers_node_sequence = self.node_sequence[start:end]

        if not self.drone_routes or not self.drone_positions or len(self.drone_routes) == 0 or len(
                self.drone_positions) == 0:
            # Return empty list
            return []

        # TODO:: Maybe remove??
        customers_node_sequence.insert(0, 0)
        customers_node_sequence.append(0)

        drone_route = self.drone_routes[truck_drone_pair_id]
        drone_launch_index = self.drone_positions[truck_drone_pair_id]
        interceptions = self.interceptions[truck_drone_pair_id]

        for i in range(len(drone_route)):
            route = []
            if drone_launch_index[i] == 0:
                route.append(self.tsp.locations[0])  # Launch drone from depot node
            else:
                launch_index = drone_launch_index[i] if drone_launch_index[i] > 0 else drone_launch_index[i] - 1
                route.append(self.tsp.locations[customers_node_sequence[launch_index]])  # Add launch node

            route.append(self.tsp.locations[drone_route[i]])  # Add delivery node

            # Add interception, if interception exists then move to next node route
            if interceptions and all(coord > 0 for coord in interceptions[i]):
                route.append(interceptions[i])
                drone_flying_route.append(route)
                continue
            else:
                if i == len(drone_route) - 1:  # If i == last drone route, send drone to depot
                    route.append(self.tsp.locations[0])
                    drone_flying_route.append(route)
                    continue

            # Only add next location if more than 1 node is to be visited by the drone
            if len(drone_route) > 1:
                route.append(self.tsp.locations[drone_launch_index[i]])  # Add next truck node
            drone_flying_route.append(route)

        return drone_flying_route

    def str_chromosome(self):
        """Convert solution chromosome to string
        :return chromosome:  value of the solution chromosome
        """
        c_str = ""
        for c in self.chromosome:
            c_str += str(c) + ":"

        c_str = c_str[:-1]  # Remove last colon
        return c_str

    def log(self, message) -> None:
        """Log message if verbose allowed
        :param message: Message to be logged
        """
        if self.tsp.verbose:
            print(message)
