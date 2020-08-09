import os

def run(command: str, config_file: str, temperature: float, more_args: str = ""):
    os.system(f"ipython Verify.py -- --command {command} --config_file ../launches/igor/{config_file} --softmax_temperature {temperature} {more_args}")

#config = ("acyclic_conveyor_energy_test.yaml", 1.5)   # Mukhutdinov
#config = ("conveyor_cyclic_energy_test.yaml", 1.5)    # Very simple fictitious graph with cycle
#config = ("conveyor_cyclic2_energy_test.yaml", 1.5)   # A complication of the previous example
#config = ("tarau2010.yaml", 1.5)                       # Fictitious graph from the literature
config = ("johnstone2010.yaml", 3.0)                  # Almost real graph from the literature

#command = "deterministic_test"
#command = "embedding_adversarial"
command = "q_adversarial"
#command = "compare"

run(command, config[0], config[1],
    more_args=""
    #more_args="--skip_graphviz"
    #more_args="--skip_graphviz --force_pretrain --force_train"
    )
