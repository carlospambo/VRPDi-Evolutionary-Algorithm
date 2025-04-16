from enum import Enum

# Constants
OUTPUT_PATH = "data/output/{folder}/{filename}"


class ProblemType(Enum):
    VRPD = "vrpdi"
    VRPD_CLUSTER = "vrpdi_cluster"
    VRPD_CLUSTER_ENDURANCE = "vrpdi_cluster_drone_endurance"
    VRP = "vrp"
    VRP_CLUSTER = "vrp_cluster"

    def is_cluster(self) -> bool:
        if self in [self.VRPD_CLUSTER, self.VRPD_CLUSTER_ENDURANCE, self.VRP_CLUSTER]:
            return True
        return False

    def is_endurance(self) -> bool:
        if self in [self.VRPD_CLUSTER_ENDURANCE]:
            return True
        return False

    def is_vrp(self) -> bool:
        if self in [self.VRP, self.VRP_CLUSTER]:
            return True
        return False


class Feasibility(Enum):
    FEASIBLE = "feasible"
    INFEASIBLE_REPEATED_DRONES = "infeasible_repeated_drone_deliveries"
    INFEASIBLE_DRONE_ENDURANCE = "infeasible_drone_endurance"


class ObjectiveFunction(Enum):
    DISTANCE = 'distance'
    DELIVERY_TIME_MEAN = 'delivery_time_mean'
    DELIVERY_TIME_MAX = 'delivery_time_max'


class PlottingObject(Enum):
    DRONE = 'DRONE'
    TRUCK = 'TRUCK'
    CUSTOMER_NODE = 'CUSTOMER_NODE'
    INTERCEPTION = 'INTERCEPTION'
    LEGEND = 'LEGEND'
    ALL = 'ALL'
