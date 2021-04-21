from ..base import *
from .link_state import *
from ...constants import DQNROUTE_LOGGER
from ...messages import *
from ...memory import *
from ...utils import *
from ...networks import *

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from dqnroute.networks.actor_critic_networks import PPOActor


class PackageHistory:
    epoch_size = 40

    routers = defaultdict(dict)
    rewards = defaultdict(list)
    log_probs = defaultdict(list)

    finished_packages = set()
    started_packages = set()

    @staticmethod
    def addToHistory(pkg: Package, router, reward: float, log_prob):
        assert pkg.id not in PackageHistory.finished_packages

        PackageHistory.routers[pkg.id][router.id] = router
        PackageHistory.rewards[pkg.id].append(reward)
        PackageHistory.log_probs[pkg.id].append(log_prob)

    @staticmethod
    def finishHistory(pkg: Package):
        PackageHistory.finished_packages.add(pkg.id)

    @staticmethod
    def learn():
        # print('Learn')
        eps = 1e-8
        gamma = 0.99

        all_routers_needed = dict()
        # print(PackageHistory.finished_packages)
        # print(PackageHistory.started_packages)
        # print(PackageHistory.finished_packages & PackageHistory.started_packages)
        for package_idx in PackageHistory.finished_packages & PackageHistory.started_packages:
            for router in PackageHistory.routers[package_idx].values():
                all_routers_needed[router.id] = router

        # print(list(all_routers_needed.keys()))
        for router in all_routers_needed.values():
            router.actor.optimizer.zero_grad()

        for package_idx in PackageHistory.finished_packages & PackageHistory.started_packages:
            rewards_package = PackageHistory.rewards[package_idx]
            log_probs_package = PackageHistory.log_probs[package_idx]

            R = 0
            policy_losses = []
            returns = []

            for r in rewards_package[::-1]:
                R = r + gamma * R
                returns.insert(0, R)

            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + eps)

            for log_prob, R in zip(log_probs_package, returns):
                policy_losses.append(-log_prob * R)

            for policy_loss in policy_losses:
                policy_loss.backward()

        for router in all_routers_needed.values():
            router.actor.optimizer.step()
            # pass

        PackageHistory.routers = defaultdict(dict)
        PackageHistory.rewards = defaultdict(list)
        PackageHistory.log_probs = defaultdict(list)

        PackageHistory.finished_packages = set()
        PackageHistory.started_packages = set()


class SharedActorStorage:
    ACTOR_INSTANCE = None
    MEMORY_INSTANCE = None
    PROCESSED_NODES = 0

    @staticmethod
    def load_actor(
            actor_loader,
            memory_loader,
            no_nodes: int,
            actor_params: dict,
            memory_params: dict
    ):
        if SharedActorStorage.INSTANCE is None:
            SharedActorStorage.INSTANCE = actor_loader()(**actor_params)
            SharedActorStorage.MEMORY_INSTANCE = memory_loader()(**memory_params)
        SharedActorStorage.PROCESSED_NODES += 1

        actor = SharedActorStorage.ACTOR_INSTANCE
        memory = SharedActorStorage.MEMORY_INSTANCE

        if SharedActorStorage.PROCESSED_NODES == no_nodes:
            # all nodes have been processes
            # prepare this class for possible reuse
            SharedActorStorage.INSTANCE = None
            SharedActorStorage.MEMORY_INSTANCE = None
            SharedActorStorage.PROCESSED_NODES = 0

        return actor, memory


class Reinforce(LinkStateRouter, RewardAgent):
    def __init__(
            self,
            distance_function: str,
            nodes: List[AgentId],
            embedding: Union[dict, Embedding],
            edges_num: int,
            actor: dict,
            max_act_time=None,
            additional_inputs=[],
            dir_with_models: str = '',
            load_filename: str = None,
            use_single_network: bool = False,  # TODO implement
            **kwargs
    ):
        super(Reinforce, self).__init__(**kwargs)

        self.distance_function = get_distance_function(distance_function)
        self.nodes = nodes
        self.init_edges_num = edges_num
        self.max_act_time = max_act_time
        self.additional_inputs = additional_inputs
        self.prev_num_nodes = 0  # TODO ???? why 0
        self.prev_num_edges = 0
        self.network_initialized = False
        self.use_single_network = use_single_network

        if type(embedding) == dict:
            self.embedding = get_embedding(**embedding)
        else:
            self.embedding = embedding

        # Init actor
        if use_single_network:
            raise NotImplementedError()
            actor, memory = SharedActorStorage.load_actor()
        else:
            # Create net architecture
            actor_args = {
                'scope': dir_with_models,
                'embedding_dim': embedding['dim']
            }
            actor_args = dict(**actor, **actor_args)
            actor_model = PPOActor(**actor_args)
            # Init net weights
            if load_filename is not None:
                # Get pretrained net from file
                actor_model.change_label(load_filename)
                # actor_model._label = load_filename
                actor_model.restore()
            else:
                # Create net from scratch
                actor_model.init_xavier()

            self.actor = actor_model
            self.memory = ReinforceMemory()

        # TODO tune params
        self.horizon = 64
        self.discount_factor = 0.99  # gamma
        self.eps = 1e-6

    def route(self, sender: AgentId, pkg: Package, allowed_nbrs: List[AgentId]) -> Tuple[AgentId, List[Message]]:
        if self.max_act_time is not None and self.env.time() > self.max_act_time:
            return super().route(sender, pkg, allowed_nbrs)
        else:
            to, addr_idx, dst_idx, action_idx, action_log_prob = self._act(pkg, allowed_nbrs)
            reward = self.registerResentPkg(
                pkg,
                addr_idx=addr_idx,
                dst_idx=dst_idx,
                action_idx=action_idx,
                action_log_prob=action_log_prob,
                allowed_neighbours=allowed_nbrs
            )
            return to, [OutMessage(self.id, sender, reward)] if sender[0] != 'world' else []

    def handleMsgFrom(self, sender: AgentId, msg: Message) -> List[WorldEvent]:
        if isinstance(msg, RewardMsg):
            addr_idx, dst_idx, action_idx, action_log_prob, allowed_neighbours, reward = self.receiveReward(msg)
            # print(f'Pkg {msg.pkg.id, msg.pkg.dst[1]}. {self.id} -> {allowed_neighbours[action_idx]}, Sender: {sender}, Time: {self.env.time()}')
            # self.memory.add((addr_idx, dst_idx, action_idx, action_log_prob, allowed_neighbours, reward))

            PackageHistory.addToHistory(msg.pkg, self, reward, action_log_prob)

            if len(PackageHistory.finished_packages) > 128:
                # PackageHistory.learn()
                pass

            return []
        else:
            return super().handleMsgFrom(sender, msg)

    def _act(self, pkg: Package, allowed_nbrs: List[AgentId]):
        allowed_nbrs_emb = list(map(lambda neighbour: self._nodeRepr(neighbour[1]), allowed_nbrs))
        allowed_nbrs_emb_tensor = torch.FloatTensor(allowed_nbrs_emb)
        addr_idx = self.id[1]
        dst_idx = pkg.dst[1]
        addr_emb = torch.FloatTensor(self._nodeRepr(addr_idx))
        dst_emb = torch.FloatTensor(self._nodeRepr(dst_idx))

        # 1. Actor generates next embedding
        predicted_next_emb = self._actorPredict(addr_emb, dst_emb)

        # 2. Compute distances from next embedding (see step 1.) and allowed neighbours
        dist_to_nbrs = self.distance_function(allowed_nbrs_emb_tensor, predicted_next_emb)
        nbrs_prob = Categorical(F.softmax(1 / (dist_to_nbrs + self.eps), dim=0))
        # TODO debug

        # 3. Get sample from allowed neighbours based on probability
        next_nbr_idx = nbrs_prob.sample()
        action_log_prob = nbrs_prob.log_prob(next_nbr_idx)
        action_idx = next_nbr_idx.item()

        to = allowed_nbrs[action_idx]
        return to, addr_idx, dst_idx, action_idx, action_log_prob

    def _actorPredict(self, addr_emb, dst_emb):
        return self.actor.forward(addr_emb, dst_emb)

    def networkStateChanged(self):
        num_nodes = len(self.network.nodes)
        num_edges = len(self.network.edges)

        if not self.network_initialized and num_nodes == len(self.nodes) and num_edges == self.init_edges_num:
            self.network_initialized = True

        if self.network_initialized and (num_edges != self.prev_num_edges or num_nodes != self.prev_num_nodes):
            self.prev_num_nodes = num_nodes
            self.prev_num_edges = num_edges
            self.embedding.fit(self.network, weight=self.edge_weight)
            # self.log(pprint.pformat(self.embedding._X), force=self.id[1] == 0)

    def _nodeRepr(self, node):
        return self.embedding.transform(node).astype(np.float32)


class ReinforceNetwork(NetworkRewardAgent, Reinforce):
    pass


class ReinforceConveyor(LSConveyorMixin, ConveyorRewardAgent, Reinforce):
    def registerResentPkg(self, pkg: Package, **kwargs) -> RewardMsg:
        addr_idx = kwargs['addr_idx']
        dst_idx = kwargs['dst_idx']
        action_idx = kwargs['action_idx']
        action_log_prob = kwargs['action_log_prob']
        allowed_neighbours = kwargs['allowed_neighbours']

        reward_data = self._getRewardData(pkg, None)

        self._pending_pkgs[pkg.id] = (addr_idx, dst_idx, action_idx, action_log_prob, allowed_neighbours, reward_data)

        return self._mkReward(pkg, reward_data)

    def _mkReward(self, bag: Bag, reward_data) -> ConveyorRewardMsg:
        # Q_estimate is ignored in our setting
        time_processed, energy_gap = reward_data
        return ConveyorRewardMsg(self.id, bag, 0, time_processed, energy_gap)

    def receiveReward(self, msg: RewardMsg):
        addr_idx, dst_idx, action_idx, action_log_prob, allowed_neighbours, old_reward_data = \
            self._pending_pkgs.pop(msg.pkg.id)

        reward = self._computeReward(msg, old_reward_data)
        return addr_idx, dst_idx, action_idx, action_log_prob, allowed_neighbours, reward

    def _computeReward(self, msg: ConveyorRewardMsg, old_reward_data):
        time_sent, _ = old_reward_data
        time_processed, energy_gap = msg.reward_data
        time_gap = time_processed - time_sent
        return -(time_gap + self._e_weight * energy_gap)
