import os
import time

def run(command: str, config_file: str, temperature: float, cost_bound: float, more_args: str = ""):
    os.system(f"ipython Run.py -- ../launches/igor/{config_file} --command {command}  --softmax_temperature {temperature} --cost_bound {cost_bound} --marabou_path ../../Marabou/build/Marabou {more_args}")

""" The original example from D. Mukhutdinov. """
#config = "acyclic_conveyor_energy_test.yaml", 1.5, 43.0 # Energy test
#config = "acyclic_conveyor_break_test.yaml",  1.5, 43.0 # Conveyor break test

""" Example from A. Tarau et al. Model-Based Control for Route Choice in Automated Baggage Handling Systems. """
#config = "tarau2010.yaml", 15.0, 972  # Original version (adapted)
config = "tarau2010_modified_common.yaml", 15.0, 972 # Mofified version with some conveyor sections extended
#config = "tarau2010_modified_break_test.yaml", 15.0, 972 # The same modified graph, conveyor break test

""" Example from M.P. Johnstone. Simulation-based learning for control of complex conveyor networks. """
#config = "johnstone2010.yaml", 3.0, 100000 # Adapted version

""" Some artififial examples of graph with cycles for smoke testing. """
#config = "conveyor_cyclic_energy_test.yaml", 1.5, 100000    # Very simple fictitious graph with cycle
#config = "conveyor_cyclic2_energy_test.yaml", 1.5, 12430.0  # A complication of the previous example

""" Select the command: """
#command, command_args = "run", ""
command, command_args = "compare", ""
#command, command_args = "deterministic_test", ""
#command, command_args = "embedding_adversarial_search", "--input_eps_l_inf 0.1"
#command, command_args = "embedding_adversarial_verification", "--output_max_delta_q 3.6 --input_eps_l_inf 0.1 --output_max_delta_p 0.00001"
#command, command_args = "embedding_adversarial_full_verification", "--input_eps_l_inf 0.001"
#command, command_args = "q_adversarial", ""
#command, command_args = "compute_expected_cost", ""
#command, command_args = "q_adversarial_lipschitz", ""

if command_args != "":
    command_args += " "

start_time = time.time()
run(command, config[0], config[1], config[2], more_args=command_args+(
    ""
    #"--skip_graphviz --verification_lr 0.001"
    #"--force_train"
    #"--skip_graphviz"
    "--force_pretrain --force_train"
    #"--skip_graphviz --force_pretrain --force_train"
    #"--skip_graphviz --force_train"
))
print(f"Elapsed time: {time.time() - start_time:.3f} s")