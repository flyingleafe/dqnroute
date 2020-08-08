import os

def run(command: str, config_file: str, more_args: str = ""):
    os.system(f"ipython Verify.py -- --command {command} --config_file ../launches/igor/{config_file} {more_args}")

#config_file = "acyclic_conveyor_energy_test.yaml"   # Mukhutdinov
#config_file = "conveyor_cyclic_energy_test.yaml"    # Very simple fictitious graph with cycle
#config_file = "conveyor_cyclic2_energy_test.yaml"   # A complication of the previous example
config_file = "tarau2010.yaml"                       # Fictitious graph from the literature
#config_file = "johnstone2010.yaml"                  # Almost real graph from the literature

#command = "deterministic_test"
#command = "embedding_adversarial"
command = "q_adversarial"
#command = "compare"

run(command, config_file,
    more_args="--force_pretrain --force_train"
    )
