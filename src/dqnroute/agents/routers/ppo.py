from typing import List, Tuple, Dict, Union

from dqnroute.networks.ppo_actor_critic_networks import PPOActor, PPOCritic

from ..base import *
from .link_state import *
from ...constants import DQNROUTE_LOGGER
from ...messages import *
from ...memory import *
from ...utils import *

from ...networks import *

import numpy as np
import torch
import torch as T
import torch.nn.functional as F

from torch.distributions.categorical import Categorical


class PPORouterEmb(LinkStateRouter, RewardAgent):
    def __init__(
            self,
            distance_function: str,
            nodes: List[AgentId],
            embedding: Union[dict, Embedding],
            edges_num: int,
            actor: dict,
            critic: dict,
            max_act_time=None,
            additional_inputs=[],
            dir_with_models: str = '',
            actor_load_filename: str = None,
            critic_load_filename: str = None,
            use_single_network: bool = False,  # TODO implement
            actor_model=None,
            critic_model=None,
            **kwargs
    ):
        super(PPORouterEmb, self).__init__(**kwargs)

        self.distance_function = get_distance_function(distance_function)
        self.nodes = nodes
        self.init_edges_num = edges_num
        self.max_act_time = max_act_time
        self.additional_inputs = additional_inputs
        self.prev_num_nodes = 0
        self.prev_num_edges = 0
        self.network_initialized = False

        if type(embedding) == dict:
            self.embedding = get_embedding(**embedding)
        else:
            self.embedding = embedding

        # Init actor
        if actor_model is None:
            actor_args = {
                'scope': dir_with_models,
                'embedding_dim': embedding['dim']
            }
            actor_args = dict(**actor, **actor_args)
            actor_model = PPOActor(**actor_args)

            if actor_load_filename is not None:
                actor_model._label = actor_load_filename
                actor_model.restore()
            else:
                actor_model.init_xavier()
        self.actor = actor_model

        # Init critic
        if critic_model is None:
            critic_args = {
                'scope': dir_with_models,
                'embedding_dim': embedding['dim']
            }
            critic_args = dict(**critic, **critic_args)
            critic_model = PPOCritic(**critic_args)

            if critic_load_filename is not None:
                critic_model._label = critic_load_filename
                critic_model.restore()
            else:
                critic_model.init_xavier()
        self.critic = critic_model

        # TODO tune params
        # Parameters from the paper for Atari
        # See https://arxiv.org/pdf/1707.06347.pdf
        self.horizon = 64  # you can try 2048
        self.n_epoch = 16
        self.minibatch_size = 16
        self.discount_factor = 0.99  # gamma
        self.GAE_parameter = 0.95  # lambda
        self.ratio_clip = 0.2
        self.critic_loss_coef = 1
        self.eps = 1e-6

        self.memory = PPOMemory(self.minibatch_size)

    def route(self, sender: AgentId, pkg: Package, allowed_nbrs: List[AgentId]) -> Tuple[AgentId, List[Message]]:
        if self.max_act_time is not None and self.env.time() > self.max_act_time:
            return super().route(sender, pkg, allowed_nbrs)
        else:
            to, q_estimate, addr_idx, dst_idx, action_idx, next_nbr_log_prob, cur_v_func = \
                self._act(pkg, allowed_nbrs)

            reward = self.registerResentPkg(
                pkg, q_estimate,
                addr_idx=addr_idx,
                dst_idx=dst_idx,
                action_idx=action_idx,
                prob=next_nbr_log_prob,
                v_func=cur_v_func,
                allowed_neighbours=allowed_nbrs
            )

            return to, [OutMessage(self.id, sender, reward)] if sender[0] != 'world' else []

    def handleMsgFrom(self, sender: AgentId, msg: Message) -> List[WorldEvent]:
        if isinstance(msg, RewardMsg):
            addr_idx, dst_idx, action_idx, prob, q_estimate, allowed_neighbours, reward, v_func = self.receiveReward(msg)
            state = addr_idx, dst_idx, action_idx, prob, q_estimate, allowed_neighbours, reward, v_func
            self.memory.add(state)

            if len(self.memory) >= self.horizon:
                self._learn()
                # pass
            return []
        else:
            return super().handleMsgFrom(sender, msg)

    def _makeBrain(self, actor_config: dict, critic_config: dict):
        actor = PPOActor(**actor_config)
        critic = PPOCritic(**critic_config)
        return actor, critic

    def _act(self, pkg: Package, allowed_nbrs: List[AgentId]):
        # 0. Init basic inputs
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
        nbrs_prob = torch.distributions.Categorical(F.softmax(1 / (dist_to_nbrs + self.eps), dim=0))

        # 3. Get sample from allowed neighbours based on probability
        next_nbr_idx = nbrs_prob.sample()
        next_nbr_log_prob = nbrs_prob.log_prob(next_nbr_idx)
        action_idx = next_nbr_idx.item()

        next_nbr_emb = torch.FloatTensor(self._nodeRepr(next_nbr_idx))

        # 4. Compute estimated current node v-function
        cur_v_func = self._criticPredict(addr_emb, dst_emb).squeeze().item()
        q_estimate = self._criticPredict(next_nbr_emb, dst_emb).squeeze().item()

        # nbrs_v_func = \
        #     self._criticPredict(
        #         allowed_nbrs_emb_tensor,
        #         torch.matmul(torch.ones(allowed_nbrs_emb_tensor.shape[0], 1), dst_emb.unsqueeze(0))
        #     )

        # q_estimate = torch.matmul(
        #     nbrs_prob.probs.unsqueeze(0),
        #     nbrs_v_func
        # ).squeeze().item()

        to = allowed_nbrs[action_idx]
        return to, q_estimate, addr_idx, dst_idx, action_idx, next_nbr_log_prob, cur_v_func

    def _actorPredict(self, addr_emb, dst_emb):
        self.actor.eval()
        return self.actor.forward(addr_emb, dst_emb).clone().detach()

    def _criticPredict(self, addr_emb, dst_emb):
        self.critic.eval()
        return self.critic.forward(addr_emb, dst_emb).clone().detach()

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

    def _learn(self):
        for _ in range(self.n_epoch):
            # Get data from memory
            addr_idx_memory, \
                dst_idx_memory, \
                action_idx_memory, \
                probs_memory, \
                q_estimates_memory, \
                neighbours_memory, \
                rewards_memory, \
                values_memory, \
                batches = self.memory.sample(None)

            # Create embeddings from indices
            addr_emb = np.array(list(map(lambda addr_idx: self._nodeRepr(addr_idx), addr_idx_memory)))
            dst_emb = np.array(list(map(lambda dst_idx: self._nodeRepr(dst_idx), dst_idx_memory)))

            neighbours_emb = []
            for neighbours_sample in neighbours_memory:
                neighbours_emb_sample = \
                    np.array(list(map(lambda neighbour_idx: self._nodeRepr(neighbour_idx[1]), neighbours_sample)))
                neighbours_emb.append(neighbours_emb_sample)

            # Start batches processing
            for batch_idxs in batches:
                # Turn on training option
                self.actor.train()
                self.critic.train()
                # Zero grad
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()

                actor_losses = []
                critic_losses = []

                # Get batch data and convert to tensors
                addr_emb_batch = torch.FloatTensor(addr_emb[batch_idxs])
                dst_emb_batch = torch.FloatTensor(dst_emb[batch_idxs])
                actions_idxs_batch = torch.tensor(action_idx_memory[batch_idxs], dtype=torch.int64)
                probs_batch = torch.FloatTensor(probs_memory[batch_idxs])
                q_estimates_batch = torch.FloatTensor(q_estimates_memory[batch_idxs])

                neighbours_emb_batch = []
                for batch_idx in batch_idxs:
                    neighbours_emb_batch.append(neighbours_emb[batch_idx])

                rewards_batch = torch.FloatTensor(rewards_memory[batch_idxs])
                values_batch = torch.FloatTensor(values_memory[batch_idxs])

                # Process every sample one by one (different neighbours count issue)
                for addr_emb_sample, \
                    dst_emb_sample, \
                    action_idx_sample, \
                    old_prob_sample, \
                    q_estimate_sample, \
                    nbrs_emb_sample, \
                    reward_sample, \
                    old_v_func_sample in \
                        zip(addr_emb_batch,
                            dst_emb_batch,
                            actions_idxs_batch,
                            probs_batch,
                            q_estimates_batch,
                            neighbours_emb_batch,
                            rewards_batch,
                            values_batch):
                    # Actor generates next embedding
                    next_emb = self.actor.forward(addr_emb_sample, dst_emb_sample)

                    # Critic generates current value function
                    cur_v_func = self.critic.forward(addr_emb_sample, dst_emb_sample)

                    # Create probabilities for each allowed neighbour
                    nbrs_emb_sample_torch = torch.FloatTensor(nbrs_emb_sample)
                    distance_to_nbrs = self.distance_function(nbrs_emb_sample_torch, next_emb)
                    nbrs_prob = Categorical(F.softmax(1 / (distance_to_nbrs + self.eps), dim=0))

                    # Get new probability of memory action
                    new_prob = nbrs_prob.log_prob(action_idx_sample)

                    # Compute 1-step advantage
                    advantage = reward_sample + self.discount_factor * q_estimate_sample - old_v_func_sample

                    # Create probability ratio
                    prob_ratio = new_prob.exp() / old_prob_sample.exp()
                    weighted_prob = advantage * prob_ratio
                    weighted_clipped_prob = T.clamp(prob_ratio, 1 - self.ratio_clip, 1 + self.ratio_clip) * advantage

                    # Compute TODO ???
                    returns = reward_sample + self.discount_factor * q_estimate_sample

                    # Compute and propagate
                    actor_loss = -torch.min(weighted_prob, weighted_clipped_prob)
                    critic_loss = (returns - cur_v_func) ** 2
                    actor_loss.backward()
                    critic_loss.backward()

                    actor_losses.append(actor_loss.item())
                    critic_losses.append(critic_loss.item())

                # Step through whole batch
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()


# TODO add another component for loss (see message history)


class PPORouterEmbNetwork(NetworkRewardAgent, PPORouterEmb):
    pass


class PPORouterEmbConveyor(LSConveyorMixin, ConveyorRewardAgent, PPORouterEmb):
    def registerResentPkg(self, pkg: Package, q_estimate: float, **kwargs) -> RewardMsg:
        addr_idx = kwargs['addr_idx']
        dst_idx = kwargs['dst_idx']
        action_idx = kwargs['action_idx']
        prob = kwargs['prob']
        v_func = kwargs['v_func']
        allowed_neighbours = kwargs['allowed_neighbours']

        reward_data = self._getRewardData(pkg, None)
        self._pending_pkgs[pkg.id] = \
            (addr_idx, dst_idx, action_idx, prob, q_estimate, allowed_neighbours, reward_data, v_func)

        return self._mkReward(pkg, q_estimate, reward_data)

    def receiveReward(self, msg: RewardMsg):
        addr_idx, dst_idx, action_idx, prob, q_estimate, allowed_neighbours, old_reward_data, v_func = \
            self._pending_pkgs.pop(msg.pkg.id)

        reward = self._computeReward(msg, old_reward_data)
        return addr_idx, dst_idx, action_idx, prob, q_estimate, allowed_neighbours, reward, v_func

    def _computeReward(self, msg: ConveyorRewardMsg, old_reward_data):
        time_sent, _ = old_reward_data
        time_processed, energy_gap = msg.reward_data
        time_gap = time_processed - time_sent
        return -time_gap
        # return -(time_gap + self._e_weight * energy_gap)
