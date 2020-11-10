import os
import time

def run(command: str, config_file: str, temperature: float, cost_bound: float, more_args: str = ""):
    os.system(f"ipython Verify.py -- --command {command} --config_file ../launches/igor/{config_file} --softmax_temperature {temperature} --cost_bound {cost_bound} --marabou_path ../../Marabou/build/Marabou {more_args}")

#config = "acyclic_conveyor_energy_test.yaml", 1.5, 43.0 # Mukhutdinov, energy test
#config = "acyclic_conveyor_break_test.yaml",  1.5, 43.0 # Mukhutdinov, conveyor break test

#config = "conveyor_cyclic_energy_test.yaml", 1.5, 100000    # Very simple fictitious graph with cycle
#config = "conveyor_cyclic2_energy_test.yaml", 1.5, 12430.0  # A complication of the previous example

#config = "tarau2010.yaml", 30.0, 972                          # Fictitious graph from the literature
config = "tarau2010_modified_common.yaml", 15.0, 972           # The same graph, but some conveyor sections were extended, making the graph asymmetric

#config = "johnstone2010.yaml", 3.0, 100000                  # Almost real graph from the literature

#command, command_args = "deterministic_test", ""
#command, command_args = "embedding_adversarial_search", "--input_eps_l_inf 0.1"
#command, command_args = "embedding_adversarial_verification", "--output_max_delta_q 3.6 --input_eps_l_inf 0.1 --output_max_delta_p 0.00001"
#command, command_args = "embedding_adversarial_full_verification", "--input_eps_l_inf 0.001"
#command, command_args = "q_adversarial", ""
#command, command_args = "compute_expected_cost", ""
#command, command_args = "q_adversarial_lipschitz", ""
command, command_args = "compare", ""

if command_args != "":
    command_args += " "

start_time = time.time()
run(command, config[0], config[1], config[2], more_args=command_args+(
    #""
    #"--skip_graphviz --verification_lr 0.001"
    #"--force_train"
    #"--skip_graphviz"
    #"--force_pretrain --force_train"
    #"--skip_graphviz --force_pretrain --force_train"
    "--skip_graphviz --force_train"
))
print(f"Elapsed time: {time.time() - start_time:.3f} s")