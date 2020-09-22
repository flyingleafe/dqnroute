import os

def run(command: str, config_file: str, temperature: float, cost_bound: float, more_args: str = ""):
    os.system(f"ipython Verify.py -- --command {command} --config_file ../launches/igor/{config_file} --softmax_temperature {temperature} --cost_bound {cost_bound} {more_args} --marabou_path ../../Marabou/build/Marabou")

#config = ("acyclic_conveyor_energy_test.yaml", 1.5, 100000)   # Mukhutdinov
#config = ("conveyor_cyclic_energy_test.yaml", 1.5, 100000)    # Very simple fictitious graph with cycle
#config = ("conveyor_cyclic2_energy_test.yaml", 1.5, 12430.0)   # A complication of the previous example
config = ("tarau2010.yaml", 4.5, 100000)                       # Fictitious graph from the literature
#config = ("johnstone2010.yaml", 3.0, 100000)                  # Almost real graph from the literature

#command = "deterministic_test"
#command = "embedding_adversarial_search"
#command = "embedding_adversarial_verification"
#command = "q_adversarial"
#command = "q_adversarial_lipschitz"
command = "compare"

run(command, config[0], config[1], config[2],
    #more_args=""
    #more_args="--skip_graphviz --verification_lr 0.001"
    #more_args="--force_train"
    more_args="--skip_graphviz"
    #more_args="--skip_graphviz --force_pretrain --force_train"
    #more_args="--skip_graphviz --force_train"
    )
