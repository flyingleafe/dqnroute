import multiprocessing
import time
import psutil

from typing import *

from RunWrapper import run
from dqnroute.verification.exception import MarabouException


def lipschitz_verification_original(bound: float):
    run("q_adversarial_lipschitz", "original_example_graph", "original_example_settings_energy_test", bound,
        "--skip_graphviz --single_source 1 --single_sink 3 --input_max_delta_q 20 "\
        "--learning_step_indices 1,2,3,8")

def lipschitz_verification_tarau(bound: float):
    run("q_adversarial_lipschitz", "tarau2010_graph_original", "tarau2010_settings_regular", bound,
        "--skip_graphviz --single_source 0 --single_sink 1 --input_max_delta_q 10 "\
        "--learning_step_indices 17,22,23,24")

def embedding_verification_original(bound: float, epsilon: float):
    run("embedding_adversarial_full_verification", "original_example_graph",
        "original_example_settings_energy_test", bound,
        f"--skip_graphviz --single_source 1 --single_sink 3 --input_eps_l_inf {epsilon} "
        f"--linux_marabou_memory_limit_mb 12288")

def embedding_verification_tarau(bound: float, epsilon: float):
    run("embedding_adversarial_full_verification", "tarau2010_graph_original", "tarau2010_settings_regular",
        bound, f"--skip_graphviz --single_source 0 --single_sink 1 --input_eps_l_inf {epsilon} "
        f"--linux_marabou_memory_limit_mb 12288")

def killall(name: str):
    for proc in psutil.process_iter():
        if proc.name() == name:
            proc.kill()
    
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
    #for eps in [3.2, 6.4]:#[1.6]:
    
    for eps in [0., 0.01, 0.1, 0.2, 0.4, 0.8]:
        for c0 [44.0, 43.5, 43.12, 43.1, 43.0]:
            run_with_timeout(embedding_verification_original, [c0, eps], TIMEOUT)
            
    #for eps in [0., 0.01, 0.1, 0.2, 0.4, 0.8]:
    #    for c0 in [850.0, 840.0, 830.0, 820.1, 820.0]:
    #        run_with_timeout(embedding_verification_tarau, [c0, eps], TIMEOUT)
    
    #lipschitz_verification_original(43.563)
    #lipschitz_verification_original(65.616)
    #lipschitz_verification_tarau(62694.8)
    #lipschitz_verification_tarau(67000.0)
