import os
import time

def run(command: str, directory: str, config_file_graph: str, config_file_settings: str,
        cost_bound: float, more_args: str):
    """
    :param command: command to be passed to Run.py.
    :param directory: directory with the conveyor topology.
    :param config_file_graph: YAML config of the conveyor graph.
    :param config_file_settings: YAML settings of bag arrival and the routing algorithm.
    :param cost_bound: mean delivery time to check (only for verification-related commands).
    :param more_args: additional arguments to be appended to the end of the call. 
    """
    run_str = (f"ipython Run.py -- ../launches/{directory}/{config_file_graph}.yaml "
               f"../launches/{directory}/{config_file_settings}.yaml --command {command} "
               f"--cost_bound {cost_bound} --marabou_path ../../Marabou/build/Marabou {more_args}")
    print(f"Running: {run_str}")
    start_time = time.time()
    os.system(run_str)
    print(f"Elapsed time: {time.time() - start_time:.3f} s")


if __name__ == "__main__":
    """ The original example from D. Mukhutdinov. """
    config = "conveyor_topology_mukhutdinov", "original_example_graph", "original_example_settings_energy_test", 10043.10052
    #config = "conveyor_topology_mukhutdinov", "original_example_graph", "original_example_settings_break_test", 43.0

    """ Example from A. Tarau et al. Model-Based Control for Route Choice in Automated Baggage Handling Systems. """
    #config = "conveyor_topology_tarau", "tarau2010_graph_original", "tarau2010_settings_regular", 100820.04923  # Original version (adapted)
    #config = "conveyor_topology_tarau", "tarau2010_graph_modified", "tarau2010_settings_regular", 972 # Modified version with some conveyor sections extended
    #config = "conveyor_topology_tarau", "tarau2010_graph_modified", "tarau2010_settings_break_test", 972 # The same modified graph, conveyor break test

    """ Example from M.P. Johnstone. Simulation-based learning for control of complex conveyor networks. """
    #config = "conveyor_topology_johnstone", "johnstone2010_graph", "johnstone2010_settings", 100000 # Adapted version

    """ Select the command: """
    # simulate/train:
    command, command_args = "run", "--routing_algorithms=dqn_emb"
    #command, command_args = "run", "--routing_algorithms=dqn_emb,centralized_simple,link_state,simple_q"
    #command, command_args = "run", "--routing_algorithms=centralized_simple,link_state,simple_q"
    
    # formal verification:
    #command, command_args = "compute_expected_cost", ""
    #command, command_args = "embedding_adversarial_search", "--input_eps_l_inf 0.01"
    #command, command_args = "embedding_adversarial_verification", "--input_eps_l_inf 0.001"
    #command, command_args = "q_adversarial_search", ""
    #command, command_args = "q_adversarial_verification", ""

    if command_args != "":
        command_args += " "

    run(command, *config, command_args+(
        #"--skip_graphviz --verification_lr 0.001"
        #"--force_train"
        "--skip_graphviz"
        #"--skip_graphviz --single_source 1 --single_sink 3"
        #"--skip_graphviz --single_source 1 --single_sink 2"
        #"--skip_graphviz --single_source 0 --single_sink 1"
        #"--skip_graphviz --single_source 2 --single_sink 0"
        #"--force_pretrain --force_train"
        #"--skip_graphviz --force_pretrain --force_train"
        #"--skip_graphviz --force_train"
        #"--skip_graphviz --single_source 1 --single_sink 3 --input_max_delta_q 20 --learning_step_indices 1,2,3,8"
        #"--skip_graphviz --single_source 0 --single_sink 1 --input_max_delta_q 10 --learning_step_indices 17,22,23,24"
    ))
