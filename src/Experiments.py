import multiprocessing
import time
import psutil

from typing import *

from RunWrapper import run


def lipschitz_verification_original(bound: float):
    run("q_adversarial_lipschitz", "original_example_graph", "original_example_settings_energy_test", bound,
        "--skip_graphviz --single_source 1 --single_sink 3 --input_max_delta_q 20 "\
        "--learning_step_indices 1,2,3,8")

def lipschitz_verification_tarau(bound: float):
    run("q_adversarial_lipschitz", "tarau2010_graph_original", "tarau2010_settings_regular", bound,
        "--skip_graphviz --single_source 0 --single_sink 1 --input_max_delta_q 10 "\
        "--learning_step_indices 17,22,23,24")

def embedding_verification_original(bound: float, epsilon: float):
    run("embedding_adversarial_full_verification", "original_example_graph", "original_example_settings_energy_test",
        bound, f"--skip_graphviz --single_source 1 --single_sink 3 --input_eps_l_inf {epsilon}")

def embedding_verification_tarau(bound: float, epsilon: float):
    run("embedding_adversarial_full_verification", "tarau2010_graph_original", "tarau2010_settings_regular",
        bound, f"--skip_graphviz --single_source 0 --single_sink 1 --input_eps_l_inf {epsilon}")

def killall(name: str):
    print("SCANNING PROCESSES")
    for proc in psutil.process_iter():
        if proc.name() == name:
            proc.kill()
            print("MARABOU KILLED")
    
def run_with_timeout(fun: Callable, args: List, timeout_sec: int):
    print()
    print("****************************************")
    print("*************** NEW RUN ****************")
    print("****************************************")
    print(f"Call: {fun.__name__}{args}")
    print()
    
    def f():
        fun(*args)
    
    try:
        # https://stackoverflow.com/questions/492519/timeout-on-a-function-call
        p = multiprocessing.Process(target=f)
        p.start()
        p.join(timeout_sec)
        if p.is_alive():
            p.terminate()
            #p.kill() #- will work for sure, no chance for process to finish nicely however
            print("TIMEOUT")
            time.sleep(1.0)
        p.join()
    finally:
        killall("Marabou")
    
if __name__ == "__main__":
    TIMEOUT = 60 * 120
    #for eps in [0., 0.01, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]:
    #for eps in [0.8, 1.6, 3.2, 6.4]:
    for eps in [1.6, 0.8, 0.4]:
        run_with_timeout(embedding_verification_original, [45.00, eps], TIMEOUT)
    
    #lipschitz_verification_original(50.0)
    #lipschitz_verification_original(70.0)
    #lipschitz_verification_tarau(1200.0)
    #lipschitz_verification_tarau(15000.0)
    
    
