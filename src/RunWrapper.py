import os
import time

def run(command: str, config_file_graph: str, config_file_settings: str, cost_bound: float, more_args: str):
    run_str = f"ipython Run.py -- ../launches/igor/{config_file_graph}.yaml ../launches/igor/{config_file_settings}.yaml"\
              f" --command {command} --cost_bound {cost_bound} --marabou_path ../../Marabou/build/Marabou {more_args}"
    print(f"Running: {run_str}")
    start_time = time.time()
    os.system(run_str)
    print(f"Elapsed time: {time.time() - start_time:.3f} s")


if __name__ == "__main__":
    """ The original example from D. Mukhutdinov. """
    config = "original_example_graph", "original_example_settings_energy_test", 10043.10052
    #config = "original_example_graph", "original_example_settings_break_test", 43.0

    """ Example from A. Tarau et al. Model-Based Control for Route Choice in Automated Baggage Handling Systems. """
    #config = "tarau2010_graph_original", "tarau2010_settings_regular", 100820.04923  # Original version (adapted)
    #config = "tarau2010_graph_modified", "tarau2010_settings_regular", 972 # Modified version with some conveyor sections extended
    #config = "tarau2010_graph_modified", "tarau2010_settings_break_test", 972 # The same modified graph, conveyor break test

    """ Example from M.P. Johnstone. Simulation-based learning for control of complex conveyor networks. """
    #config = "johnstone2010_graph", "johnstone2010_settings", 100000 # Adapted version

    """ Some artificial examples of graph with cycles for smoke testing. """
    #config = "cyclic_example1_graph", "cyclic_example_settings", 100000  # Very simple fictitious graph with cycle
    #config = "cyclic_example2_graph", "cyclic_example_settings", 12430.0 # A complication of the previous example

    """ Select the command: """
    #command, command_args = "run", ""
    #command, command_args = "compare", ""
    #command, command_args = "deterministic_test", ""
    command, command_args = "embedding_adversarial_search", "--input_eps_l_inf 0.2"
    #command, command_args = "embedding_adversarial_verification", "--output_max_delta_q 3.6 --input_eps_l_inf 0.1 --output_max_delta_p 0.00001"
    #command, command_args = "embedding_adversarial_full_verification", "--input_eps_l_inf 0.001"
    #command, command_args = "q_adversarial", ""
    #command, command_args = "compute_expected_cost", ""
    #command, command_args = "q_adversarial_lipschitz", ""

    if command_args != "":
        command_args += " "

    run(command, *config, command_args+(
        ""
        #"--skip_graphviz --verification_lr 0.001"
        #"--force_train"
        #"--skip_graphviz"
        #"--skip_graphviz --single_source 1 --single_sink 3"
        "--skip_graphviz --single_source 1 --single_sink 2"
        #"--skip_graphviz --single_source 0 --single_sink 1"
        #"--skip_graphviz --single_source 2 --single_sink 0"
        #"--force_pretrain --force_train"
        #"--skip_graphviz --force_pretrain --force_train"
        #"--skip_graphviz --force_train"
        #"--skip_graphviz --single_source 1 --single_sink 3 --input_max_delta_q 20 --learning_step_indices 1,2,3,8"
        #"--skip_graphviz --single_source 0 --single_sink 1 --input_max_delta_q 10 --learning_step_indices 17,22,23,24"
    ))
