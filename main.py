import time

from ga import GeneticAlgorithm
from tsp import TSP
from configs import ProblemType, ObjectiveFunction, PlottingObject


# Main
def execute(
        tsp_inst: TSP,
        experiments: int = 1,
        save_logs: bool = True,
        show_logs: bool = False,
        fig_objects=None
):
    """Run experiment
    :param tsp_inst: TSP Instance
    :param experiments: The number of executions to run
    :param save_logs: Whether to save logs of the experiment or not
    :param show_logs: Whether to display the logs of the experiment or not
    :param fig_objects: Object figures
    :return dictionary: Genetic Algorithm object and CPU run time
    """
    if fig_objects is None:
        fig_objects = [PlottingObject.DRONE, PlottingObject.INTERCEPTION, PlottingObject.TRUCK]

    ga = GeneticAlgorithm(tsp_inst)
    o_filename = tsp_inst.output_filename()
    tsp_inst.write_to_file(o_filename)
    lines = tsp_inst.count_lines(o_filename) + 1
    for exp_no in range(1, experiments + 1):

        start_time = time.time()
        ga.run(save_logs, exp_no)
        cpu_time = time.time() - start_time
        chromosome = ga.best.str_chromosome()

        msg = f"{lines},{ga.best.total_distance},{ga.best.total_delivery_time},"
        msg += f"{ga.best.total_waiting_time},{cpu_time},{ga.best.generations},{chromosome}"
        tsp_inst.write_to_file(o_filename, msg)

        # Plot VRPDi graph
        if show_logs:
            msg = f"Dataset: {tsp_inst.filename}, Problem Type: {tsp_inst.problem_type.value.upper()}\n"
            msg += f"Interactions: {tsp_inst.no_iterations}, Population: {tsp_inst.population_size}, "
            msg += f"Elite Population: {tsp_inst.elite_population_size}, Mutation: {tsp_inst.mutation_rate}\n"
            msg += f"CPU Time: {cpu_time :.2f}s, Distance: {ga.best.total_distance :.2f}, "
            msg += f"Delivery Time: {ga.best.total_delivery_time :.2f}, Waiting Time: {ga.best.total_waiting_time :.2f}, "
            msg += f"Generations: {ga.best.generations}\nChromosome: {ga.best.chromosome}\n"
            print(msg)

        w, h = min(int(tsp_inst.no_nodes * .6) + 2, 21), min(int(tsp_inst.no_nodes * .5), 13)
        ga.plot(fig_size=(w, h), fig_objects=fig_objects, experiment_no=exp_no)
        lines += 1

    return ga


if __name__ == '__main__':

    # Define the datasets names
    files = ['uniform-51-n10']

    for file in files:
        for problem_type in [ProblemType.VRP, ProblemType.VRPD]:
            tsp = TSP(file, problem_type=problem_type, objective_function=ObjectiveFunction.DELIVERY_TIME_MAX)

            # Run experiment
            execute(tsp)
