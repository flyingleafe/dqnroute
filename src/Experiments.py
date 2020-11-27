from RunWrapper import run

def lipschitz_verification_original(bound: float):
    run("q_adversarial_lipschitz", "original_example_graph", "original_example_settings_energy_test", bound, "--skip_graphviz --single_source 1 --single_sink 3 --input_max_delta_q 20 --learning_step_indices 1,2,3,8")

def lipschitz_verification_tarau(bound: float):
    run("q_adversarial_lipschitz", "tarau2010_graph_original", "tarau2010_settings_regular", bound, "--skip_graphviz --single_source 0 --single_sink 1 --input_max_delta_q 10 --learning_step_indices 17,22,23,24")

#lipschitz_verification_original(50.0)
#lipschitz_verification_original(70.0)
#lipschitz_verification_tarau(1200.0)
lipschitz_verification_tarau(15000.0)