## 1. Preface: DQNroute

The original project [DQNroute](https://github.com/flyingleafe/dqnroute) comprises a simulation model for package delivery in computer networks, a simulation model for baggage delivery, and a reinforcement learning approach to learn a single neural network that controls routing in a distributed way. This approach is described in [[Mukhutdinov, D., Filchenkov, A., Shalyto, A., & Vyatkin, V. (2019). Multi-agent deep learning for simultaneous optimization for time and energy in distributed routing system. Future Generation Computer Systems, 94, 587-600]](https://www.sciencedirect.com/science/article/pii/S0167739X18309087?casa_token=3O7gKwF4KRAAAAAA:Ia9qKHkdtgvekRjrCL_M7U5jBFpIYCVPMUagJTf88lWfjJrv6D7zNkaJyYIPj9mculdSsbLXYhI). Beyond this publication, DQNroute is enhanced by using Laplacian Eigenmap embeddings of nodes to be adaptive to the changes in network topology.

## 2. Introduction

This project enhances [DQNroute](https://github.com/flyingleafe/dqnroute) in several ways.
The modifications concern only the baggage handling capabilities of the original project, but may potentially be adapted to the domain of computer networks.

The changes w.r.t. the original DQNroute are:

* Bugfix: a bag was processed incorrectly if it passed twice along the same conveyor. This is possible only in topology graphs with cycles.
* Bugfix: Laplacian Eigenmap and HOPE embeddings were computed nondeterministically (with different random initialization), making different nodes have different embedding matrices. The initialization was replaced by a fixed, deterministic one.
* An option that allows configuring whether the neural networks in different nodes share their parameters.
    * Configure this in YAML configuration files under settings -> router -> dqn -> use_single_neural_network (True/False). If not specified, the default is False.
    * The original version of DQNroute with Laplacian Eigenmap embeddings worked as follows: a single neural network was pretrained and copied to all the nodes, then the nodes modified their own versions of the neural network during simulation. This corresponds to use_single_neural_network = False.
    * To make all nodes execute/train the same neural network, set use_single_neural_network = True.
* Neural network verification methods. They are explained in detail in Section 6.
* The decisions of the neural network are altered to exclude routing probabilities that are very close to 0 and 1. This is done similarly to label smoothing. This is needed for the verification methods to work properly.
* A [script](/src/Run.py) (use with "--command run") that offers an easier access to the original project by performing both pretraining and training. It also visualizes topology graphs.
* New baggage handling conveyor networks/graphs:
    * [conveyor_topology_tarau](/launches/conveyor_topology_tarau) based on [[Tarau, Alina N., Bart De Schutter, and Hans Hellendoorn. "Model-based control for route choice in automated baggage handling systems." IEEE Transactions on Systems, Man, and Cybernetics, Part C (Applications and Reviews) 40.3 (2010): 341-351]](https://ieeexplore.ieee.org/abstract/document/5382550/);
    * [conveyor_topology_johnstone](/launches/conveyor_topology_johnstone) based on [[Johnstone, Michael, Doug Creighton, and Saeid Nahavandi. "Status-based routing in baggage handling systems: Searching verses learning." IEEE Transactions on Systems, Man, and Cybernetics, Part C (Applications and Reviews) 40.2 (2009): 189-200]](https://ieeexplore.ieee.org/abstract/document/5357429/).

## 3. Dependencies

* Python package dependencies are in [requirements.txt](/requirements.txt).
* Install pytorch according to the instructions on [this page](https://pytorch.org/get-started/locally/).
* You can get simpy from [here](https://pypi.org/project/simpy/). Note that simpy and sympy are completely different packages, and both are needed.
* (Optional, to get conveyor topology plots, can be disabled) Pygraphviz: [for Linux/Mac](https://anaconda.org/anaconda/pygraphviz), [for Windows](https://anaconda.org/alubbock/pygraphviz). Pygraphviz requires graphviz to be installed and available from the command line. Additional instructions for Windows:
   1. Download graphviz-2.38.msi from [here](https://graphviz.gitlab.io/_pages/Download/Download_windows.html) and install it.
   2. Download the wheel file you need from [here](http://www.lfd.uci.edu/~gohlke/pythonlibs/#pygraphviz) (for instance, if you have python **3.7** and  **windows** **x64** platform, then install "pygraphviz‑1.6‑cp37‑cp**37**m‑**win**_amd**64**.whl").
   3. Navigate to the directory to which you downloaded the wheel file.
   4. Run ```pip install <the name of the file you downloaded>```.
* (Optional, to use the "embedding_adversarial_verification" command of Run.py) You need to install [Marabou](https://github.com/NeuralNetworkVerification/Marabou). Marabou is executed as a process and you need to pass the path to the executable as a command line argument --marabou_path.

## 4. Running the project

### Using the script (preferred)

Run the script [Run.py](/src/Run.py) from the "src" directory. You can get a summary of its command line arguments by running it as follows:
```console
cd src
python Run.py -h
```

Most likely, you will need its "run" command, which, by default, will run DQNroute and several other algorithms (not based on neural networks) for comparison. Run it without arguments to get help on command line options. With them, you can configure which algorithms are executed and whether a pretrained neural network is loaded from a file. Once all the algorithms are run, this command will create delivery time and energy consumption plots. Also note that you need to provide additional configuration file(s) to this script. They are explained in more detail below in Section 5.

You can start with the examples given in [RunWrapper.py](/src/RunWrapper.py), e.g.:
```console
cd src
ipython RunWrapper.py
```

The Run.py script supports several more commands that are related to formal verification and gradient-based search of adversarial examples for DQNroute. They are explained in more detail below in Section 6.

### Using Jupyter notebooks (not maintained for some time)

This option does not provide access to verification methods, but at the moment this is the only option to run the project on the computer network routing problem.

Run the notebooks: [new_pretrain.ipynb](/notebooks/new_pretrain.ipynb), then [new_train.ipynb](/notebooks/new_train.ipynb). Running new_train.ipynb requires having a pretrained model generated by new_pretrain.ipynb for the same topology graph and embedding dimension (see scenario, emb_dim, graph_size = ... assignments). Unfortunately, the graph size needs to be entered manually. It is equal to the total number of sources, junctions, diverters and sinks.

There are more notebooks in the [notebooks](/notebooks) directory. Unfortunately, at the moment, all the notebooks may be out-of-date and may refer to missing (or moved) files.

## 5. Configuration files

The project requires YAML configuration files to be prepared. These files describe the network (conveyor or computer network), the scenario of incoming bags or packages, possible malfunctions in the network, and the (hyper)parameters of routing algorithms.

### Configuration files for the baggage handling problem

For example, see the following YAML files:

* [launches/conveyor_topology_mukhutdinov/original_example_graph.yaml](/launches/conveyor_topology_mukhutdinov/original_example_graph.yaml): contains the description of the topology graph of the conveyor network ("configuration");
* [launches/conveyor_topology_mukhutdinov/original_example_settings_break_test.yaml](/launches/conveyor_topology_mukhutdinov/original_example_settings_break_test.yaml): contains the parameters of routing algorithms ("router"), the settings of the conveyor network ("conveyor_env", "conveyor") and the scenario of incoming bags ("bags_distr"), including possible conveyor malfunctions.

The description of the topology graph was intentionally split from other settings for convenience. If you use Run.py (see above), just specify both configuration files as command line arguments. The script will concatenate all the provided files. At the moment, Jupyter notebooks will require all settings to be provided in a single file.

In the [launches](/launches) directory, you can find several example conveyor topologies with visualizations and separate README.md files.

General notes regarding supported topology networks/graphs:

* The conveyor network is described by sources, sinks, conveyors and diverters.
* It is not allowed to have a conveyor that ends and begins at the same different conveyor.
* A conveyor is not allowed to end at some earlier position of itself.
* Internally, conveyors will be split into conveyor sections, and the network will be transformed into a directed graph of *checkpoints* connected with conveyors. The checkpoints are:
    * Sources: bag arrival points.
    * Sinks: bag destinations (a fixed sink is specified for each bag and is not changed during the delivery).
    * Junctions: points where one conveyor ends and redirects its bags onto a different conveyor. A junction can join two conveyor sections (one conveyor end and another begins) or three conveyor sections (one conveyor ends and redirects the bag at a non-zero position of another conveyor).
    * Diverters: devices that can redirect each bag from the current conveyor to another conveyor.
* The aforementioned assumptions imply that the maximum indegree of a node is two and is only possible for a junction, and the maximum outdegree of a node is also two and is only possible for a diverter.
* If you have Graphviz installed (see Section 2), conveyor graphs can be visualized by Run.py.

### Configuration files for computer network routing problem

You can see examples in the [launches/conveyor_network_launches](/launches/conveyor_network_launches) directory. At the moment, configurations for the computer network routing problem are not split into two files. Topology graphs are specified directly as directed graphs.

## 6. RL_Verif: formal verification of neural networks for baggage routing

The script [Run.py](/src/Run.py) and the subpackage [verification](/src/dqnroute/verification) implement several methods of neural network analysis and verification specifically for the baggage handling problem.

The implemented features correspond to different commands (--command ...) of Run.py:

* "compute_expected_cost": computes the expected bag delivery time (EBDT), assuming that the network is frozen (not learning) during the delivery. This assumption is also used below unless specified otherwise.
* "embedding_adversarial_search": searches for adversarial examples that maximize the EBDT with respect to input node embeddings. The search is implemented with projected gradient descent (PGD).
* "embedding_adversarial_verification": verifies the adversarial robustness of the EBDT with respect to node embeddings. Verification is implemented by using the [Marabou](https://github.com/NeuralNetworkVerification/Marabou) framework. Verification may be slow.
* "q_adversarial_search": visualizes the changes in the EBDT when the network is altered by a single gradient descent step performed in a particular combination of nodes. This command can be used to approximately find the maximum of the EBDT.
* "q_adversarial_verification": verifies that the EBDT in the aforementioned circumstances does not exceed the provided bound. Verification is implemented through the estimation of Lipschitz constants of scalar-input functions.

Refer to [the slides](/rl_verif.pdf) for more information.

## 7. Overview of project structure

* [src/dqnroute/networks](/src/dqnroute/networks): definitions of neural network architectures;
* [src/dqnroute/agents/routers](/src/dqnroute/agents/routers): DQNroute and several other routing algorithms (baselines);
* [src/dqnroute/agents/conveyors](/src/dqnroute/agents/conveyors), [src/dqnroute/conveyor_model](/src/dqnroute/conveyor_model), [src/dqnroute/simulation](/src/dqnroute/simulation): implementation of the simulation models of conveyor and computer networks;
* [src/dqnroute/verification](/src/dqnroute/verification): analysis and verification approaches described in Section 6.
