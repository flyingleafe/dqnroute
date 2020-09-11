## Introduction

This project enhances [dqnroute](https://github.com/flyingleafe/dqnroute) in several ways.
The modifications concern only the baggage handling capabilities of the original project.

Original citation: [Mukhutdinov, D., Filchenkov, A., Shalyto, A., & Vyatkin, V. (2019). Multi-agent deep learning for simultaneous optimization for time and energy in distributed routing system. Future Generation Computer Systems, 94, 587-600](https://www.sciencedirect.com/science/article/pii/S0167739X18309087?casa_token=3O7gKwF4KRAAAAAA:Ia9qKHkdtgvekRjrCL_M7U5jBFpIYCVPMUagJTf88lWfjJrv6D7zNkaJyYIPj9mculdSsbLXYhI).

## Changes w.r.t. the original dqnroute

* Neural network verification methods (work in progress).
* The decisions of the neural network are altered to exclude routing probabilities that are very close to 0 and 1. This is done similarly to label smoothing.
* A [script](/src/Verify.py) that offers an easier access to the original project by performing both pretraining and training.
* [Examples](igor_dqn_pretrain.ipynb) of baggage handling topology graphs, in particular with cycles.
* A fix (?) for the bug: a bag was processed incorrectly if it passed twice along the same conveyor. This is possible only in topology graphs with cycles.

Unfortunately, some features are currently implemented with global variables due to Igor's lack of good understanding of the simulation model.

## Dependencies

The file [requirements.txt](/requirements.txt) is not up-to-date. Just install whatever is missing. You will at least need to have the following packages: torch, sympy, pygraphviz.

## Running

Option 1: run notebooks, such as [igor_dqn_pretrain.ipynb](/notebooks/igor_dqn_pretrain.ipynb) and [igor_run_simulation.ipynb](/notebooks/igor_run_simulation.ipynb).
Not really maintained for some time, may contain obsolete code. Not recommended.

Option 2: run an all-in-one verification script [Verify.py](/src/Verify.py). In particular, it contains the "compare" command, which runs simulations without any verification. This command will create delivery time and energy consumption plots. Some examples are given in [VerifyWrapper.py](/src/VerifyWrapper.py).
