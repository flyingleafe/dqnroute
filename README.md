## Preface: dqnroutre

The original project [dqnroute](https://github.com/flyingleafe/dqnroute) comprises a simulation models for package delivery in computer networks, a simulation model for baggage delivery, and a reinforcement learning approach to learn a single neural network that controls routing in a distributed way. This approach is described in [Mukhutdinov, D., Filchenkov, A., Shalyto, A., & Vyatkin, V. (2019). Multi-agent deep learning for simultaneous optimization for time and energy in distributed routing system. Future Generation Computer Systems, 94, 587-600](https://www.sciencedirect.com/science/article/pii/S0167739X18309087?casa_token=3O7gKwF4KRAAAAAA:Ia9qKHkdtgvekRjrCL_M7U5jBFpIYCVPMUagJTf88lWfjJrv6D7zNkaJyYIPj9mculdSsbLXYhI). Beyond this publication, dqnroute is enhanced by using Laplacian embeddings of nodes to be adaptive to the changes in network topology.

## Introduction

This project enhances [dqnroute](https://github.com/flyingleafe/dqnroute) in several ways.
The modifications concern only the baggage handling capabilities of the original project, but may potentially be adapted to the domain of computer networks.

The changes w.r.t. the original dqnroute are:

* Neural network verification methods (work in progress).
* The decisions of the neural network are altered to exclude routing probabilities that are very close to 0 and 1. This is done similarly to label smoothing. This is needed for the verification methods.
* A [script](/src/Verify.py) that offers an easier access to the original project by performing both pretraining and training. It also visualizes topology graphs.
* [Examples](/launches/igor) of baggage handling topology graphs, in particular with cycles:
    * [Example](/launches/igor/tarau2010.yaml) (and [visualization](/launches/igor/ConveyorGraph-Tarau2010.pdf)) based on [Tarau, Alina N., Bart De Schutter, and Hans Hellendoorn. "Model-based control for route choice in automated baggage handling systems." IEEE Transactions on Systems, Man, and Cybernetics, Part C (Applications and Reviews) 40.3 (2010): 341-351](https://ieeexplore.ieee.org/abstract/document/5382550/).
    * [Example](/launches/igor/johnstone2010.yaml) (and [visualization](/launches/igor/ConveyorGraph-Johnstone2010.pdf)) based on [Johnstone, Michael, Doug Creighton, and Saeid Nahavandi. "Status-based routing in baggage handling systems: Searching verses learning." IEEE Transactions on Systems, Man, and Cybernetics, Part C (Applications and Reviews) 40.2 (2009): 189-200](https://ieeexplore.ieee.org/abstract/document/5357429/).
* A fix (?) for the bug: a bag was processed incorrectly if it passed twice along the same conveyor. This is possible only in topology graphs with cycles.

Unfortunately, some features are currently implemented with global variables due to Igor's lack of good understanding of the simulation model.

## RL_Verif: verification of neural networks for baggage routing

The script [Verify.py](/src/Verify.py) and a subpackage [verification](/src/dqnroute/verification) implement several methods of neural network analysis and verification. This is still work in progress and may contain bugs.

The implemented features are:

* Command "embedding_adversarial_search": search for adversarial examples that maximize bag delivery time with respect to input node embeddings, assuming that the network is frozen (not learning) during the delivery. The search is implemented with projected gradient descent (PGD).
* Command "embedding_adversarial_verification": verification of the stability of the outputs of the neural network: its Q value prediction (when run for a fixed current node / neighbor / destination combination) and routing probabilities (for a fixed diverted / destination combination). Verification is implemented by using the [Marabou](https://github.com/NeuralNetworkVerification/Marabou) framework.
* Command "q_adversarial": visualization of the changes on delivery time when the network is altered by a single gradient descent step performed in a particular combination of nodes.
* Command "q_adversarial_lipschitz": verification that the delivery time in the aforementioned circumstances does not exceed the provided bound. Verification is implemented through the estimation of Lipschitz constants of scalar-input functions.

Refer to [the slides](/rl_verif.pdf) for more information.

## Dependencies

Python package dependencies are in [requirements.txt](/requirements.txt). You can get simpy from [here](https://pypi.org/project/simpy/). Note that simpy and sympy are completely different packages, and both are needed. Pygraphviz: [for Linux/Mac](https://anaconda.org/anaconda/pygraphviz), [for Windows](https://anaconda.org/alubbock/pygraphviz).

To use the "embedding_adversarial_verification" command of Verify.py, you need to install [Marabou](https://github.com/NeuralNetworkVerification/Marabou). Marabou is executed as a process and you need to pass the path to the executable as a command line argument --marabou_path.

## Running

Option 1. Run an all-in-one verification script [Verify.py](/src/Verify.py). In particular, it contains the "compare" command, which runs simulations without any verification. This command will create delivery time and energy consumption plots. Some examples are given in [VerifyWrapper.py](/src/VerifyWrapper.py). Run the scripts from the "src" directory, e.g.:
```console
cd src
ipython VerifyWrapper.py
```

Option 2 (without verification). Run notebooks: [new_pretrain.ipynb](/notebooks/new_pretrain.ipynb), then [new_train.ipynb](/notebooks/new_train.ipynb). Running new_train.ipynb requires having a pretrained model generated by new_pretrain.ipynb for the same topology graph and embedding dimension (see scenario, emb_dim, graph_size = ... assignments). Unfortunately, the graph size needs to be entered manually. It is equal to the total number of sources, junctions, diverters and sinks.
