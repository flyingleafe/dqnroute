from typing import List, Tuple, Dict, Union
from ..base import *
from .link_state import *
from ...constants import DQNROUTE_LOGGER
from ...messages import *
from ...memory import *
from ...utils import *
from ...networks import *

import os
import numpy as np
import torch
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from collections import defaultdict

# device = 'cuda:0' if T.cuda.is_available() else 'cpu'


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.neighbours = []

        self.batch_size = batch_size

    def generate_batches(self):
        # n_states = len(self.states)
        # batch_start = np.arange(0, n_states, self.batch_size)

        neighbours_count_to_idx = defaultdict(list)
        for idx, allowed_neighbours in enumerate(self.neighbours):
            neighbours_count = len(allowed_neighbours)
            neighbours_count_to_idx[neighbours_count].append(idx)

        batches = []

        for length, idxs in neighbours_count_to_idx.items():
            specific_batches_starts = np.arange(0, len(idxs), self.batch_size)

            idxs_numpy = np.array(idxs, dtype=np.int64)
            np.random.shuffle(idxs_numpy)

            specific_length_batches = [idxs_numpy[i:i + self.batch_size] for i in specific_batches_starts]

            batches.extend(specific_length_batches)

        return \
            np.array(self.states), np.array(self.actions), \
            np.array(self.probs), np.array(self.vals), \
            np.array(self.rewards), self.neighbours, \
            batches

    def store_memory(self, state, action, probs, vals, reward, neighbours):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.neighbours.append(neighbours)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.vals = []
        self.neighbours = []

    def __len__(self):
        return len(self.actions)


# baggage_destination_point, current_state -> next_state
class Actor(nn.Module):
    def __init__(self, input_shape: int, output_shape: int, learning_rate: float):
        super(Actor, self).__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.learning_rate = learning_rate

        self.checkpoint_file = os.path.join('./', 'actor_torch_ppo')

        self.fc1_dim = 64
        self.fc2_dim = 64

        self.actor = nn.Sequential(
            nn.Linear(self.input_shape, self.fc1_dim),
            nn.ReLU(),
            nn.Linear(self.fc1_dim, self.fc2_dim),
            nn.ReLU(),
            nn.Linear(self.fc2_dim, self.output_shape)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    # Works with batches
    def forward(self, actor_input):
        return self.actor(actor_input)

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


# baggage_destination_point, current_state -> value_function
class Critic(nn.Module):
    def __init__(self, input_shape: int, output_shape: int, learning_rate: float):
        super(Critic, self).__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.learning_rate = learning_rate

        self.checkpoint_file = os.path.join('./', 'actor_torch_ppo')

        self.fc1_dim = 64
        self.fc2_dim = 64

        self.critic = nn.Sequential(
            nn.Linear(self.input_shape, self.fc1_dim),
            nn.ReLU(),
            nn.Linear(self.fc1_dim, self.fc2_dim),
            nn.ReLU(),
            nn.Linear(self.fc2_dim, self.output_shape)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    # Works with batches
    def forward(self, critic_input):
        return self.critic(critic_input)

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class PPORouterEmb(LinkStateRouter, RewardAgent):
    def __init__(self, state_shape: int, distance_function: str, embedding: Union[dict, Embedding], edges_num: int,
                 nodes: List[AgentId],
                 max_act_time=None, **kwargs):
        super(PPORouterEmb, self).__init__(**kwargs)

        self.state_shape = state_shape
        self.distance_function = get_distance_function[distance_function]
        self.max_act_time = max_act_time
        self.init_edges_num = edges_num

        self.prev_num_nodes = 0
        self.prev_num_edges = 0
        self.init_edges_num = edges_num
        self.network_initialized = False

        if type(embedding) == dict:
            self.embedding = get_embedding(**embedding)
        else:
            self.embedding = embedding

        self.nodes = nodes

        # tune params
        # Parameters from the paper for Atari
        # See https://arxiv.org/pdf/1707.06347.pdf
        self.horizon = 127  # you can try 2048
        self.learning_rate = 3 * 1e-4  # add specific learning_rate for both actor and critic
        self.n_epoch = 3
        self.minibatch_size = 32
        self.discount_factor = 0.99  # gamma
        self.GAE_parameter = 0.95  # lambda
        self.policy_ratio_clip = 0.2
        self.critic_loss_coef = 1  # c_1 in loss function
        self.eps = 1e-8

        self.actor_input_shape = self.state_shape + self.state_shape  # baggage_destination_state + current_state
        self.actor_output_shape = self.state_shape  # next_state

        self.critic_input_shape = self.state_shape + self.state_shape  # baggage_destination_state + current_state
        self.critic_output_shape = 1  # value of state-function V(s)

        self.actor = Actor(self.actor_input_shape, self.actor_output_shape, self.learning_rate)
        self.critic = Critic(self.critic_input_shape, self.critic_output_shape, self.learning_rate)

        self.memory = PPOMemory(self.minibatch_size)

    # Works only with single values
    def remember(self, state, action, prob, value, reward, allowed_neighbours):
        """Saves a transition into a buffer"""
        self.memory.store_memory(state, action, prob, value, reward, allowed_neighbours)

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    # Part of routing interface
    def route(self, sender: AgentId, pkg: Package, allowed_nbrs: List[AgentId]) -> Tuple[AgentId, List[Message]]:
        if self.max_act_time is not None and self.env.time() > self.max_act_time:
            return super().route(sender, pkg, allowed_nbrs)
        else:
            to, value, actor_state, action_idx, prob = self._act(pkg, allowed_nbrs)

            reward = self.registerResentPkg(
                pkg, value, to, actor_state,
                state=actor_state,
                action_idx=action_idx,
                prob=prob,
                value=value,
                allowed_neighbours=allowed_nbrs
            )

            # print(
            #     f'Route. To: {to}, Id: {self.id}, '
            #     f'State: {actor_state}, Value: {value}, '
            #     f'Action_idx: {action_idx}, Prob: {prob}, '
            #     f'Reward: {reward}, Allowed: {allowed_nbrs}'
            # )

            return to, [OutMessage(self.id, sender, reward)] if sender[0] != 'world' else []

    # Works only with single package and list of neighbours (not for batch)
    def _act(self, pkg: Package, allowed_nbrs: List[AgentId]):
        allowed_neighbours = []
        for allowed_neighbour in allowed_nbrs:
            allowed_neighbours.append(self._nodeRepr(allowed_neighbour[1]))

        allowed_neighbours_states = torch.tensor([allowed_neighbours], dtype=torch.float32)  # Convert to batch format

        actor_state = self._getActorNNState(pkg)
        actor_state_torch = torch.tensor(actor_state)
        actor_state_torch = actor_state_torch.reshape(1, -1)  # Convert to batch format

        critic_state = self._getCriticNNState(pkg)  # Try different critic state (current or next)
        critic_state_torch = torch.tensor(critic_state)
        critic_state_torch = critic_state_torch.reshape(1, -1)  # Convert to batch format

        predicted_state = self._actorPredict(actor_state_torch)
        predicted_state = predicted_state.reshape(1, -1)  # Convert to batch format

        distance_to_neighbours = self.distance_function(predicted_state, allowed_neighbours_states)
        neighbours_distribution = Categorical(1 / (distance_to_neighbours + self.eps))

        value_function = self._criticPredict(critic_state_torch)

        action_idx = neighbours_distribution.sample()

        prob = T.squeeze(neighbours_distribution.log_prob(action_idx)).item()
        action_idx = T.squeeze(action_idx).item()
        value = T.squeeze(value_function).item()

        to = allowed_nbrs[action_idx]

        # estimate = 0  # I think it is V(s)
        # return to, estimate, actor_state, action_idx, prob, value  # ask about actor_state
        return to, value, actor_state, action_idx, prob

    # Part of routing interface
    def handleMsgFrom(self, sender: AgentId, msg: Message) -> List[WorldEvent]:
        if isinstance(msg, RewardMsg):
            saved_state, action_idx, prob, value, allowed_neighbours, action, reward = self.receiveReward(msg)

            # print(f'Handle. To: {action}, Id: {self.id}, '
            #       f'State: {saved_state[0]}, Value: {value[0]}, '
            #       f'Action_idx: {action_idx[0]}, Prob: {prob[0]}, '
            #       f'Reward: {-reward}, Allowed: {allowed_neighbours}'
            #       )

            self.remember(saved_state[0], action_idx[0], prob[0], value[0], -reward, allowed_neighbours)

            if len(self.memory) > self.horizon:
                self._learn()
            return []
        else:
            return super().handleMsgFrom(sender, msg)

    # Done (actually no changes, just copied from dqn.py)
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

    # Works only with batches
    def _actorPredict(self, actor_state: torch.Tensor) -> torch.Tensor:
        predicted_state = self.actor.forward(actor_state)
        return predicted_state

    # Works only with single package (not for batch)
    def _getActorNNState(self, pkg: Package) -> np.ndarray:
        """Returns input for actor network"""
        current_state = self._nodeRepr(self.id[1])
        destination_state = self._nodeRepr(pkg.dst[1])
        return np.concatenate((destination_state, current_state))

    # Works only with batches
    def _criticPredict(self, critic_state: torch.Tensor) -> torch.Tensor:
        predicted_v_function = self.critic.forward(critic_state)
        return predicted_v_function

    # Works only with single package (not for batch)
    def _getCriticNNState(self, pkg: Package) -> np.ndarray:
        """Returns input for critic network"""
        current_state = self._nodeRepr(self.id[1])
        destination_state = self._nodeRepr(pkg.dst[1])
        return np.concatenate((destination_state, current_state))

    # Works only with single node (not for batch)
    def _nodeRepr(self, node):
        """Transforms node into embedding"""
        return self.embedding.transform(node).astype(np.float32)

    # Update weights for actor and critic neural networks
    def _learn(self):
        """Updates actor and critic weights based on last policy history"""
        for _ in range(self.n_epoch):
            states_memory, actions_memory, \
            old_probabilities_memory, values_memory, \
            rewards_memory, allowed_neighbours_memory, \
            batches = self.memory.generate_batches()

            allowed_neighbours_emb = []
            for allowed_neighbours_sample in allowed_neighbours_memory:
                allowed_neighbours_emb_sample = \
                    list(map(lambda neighbour: self._nodeRepr(int(neighbour[1])), allowed_neighbours_sample))
                allowed_neighbours_emb.append(allowed_neighbours_emb_sample)

            advantage = np.zeros(len(rewards_memory), dtype=np.float32)

            for t in range(len(rewards_memory) - 1):
                discount = 1
                advantage_t = 0
                for k in range(t, len(rewards_memory) - 1):
                    delta_k = rewards_memory[k] + self.discount_factor * values_memory[k + 1] - values_memory[k]
                    advantage_t += discount * delta_k

                    discount *= self.discount_factor * self.GAE_parameter
                advantage[t] = advantage_t

            advantage = T.tensor(advantage)
            values = T.tensor(values_memory)

            for batch in batches:
                states = T.tensor(states_memory[batch], dtype=T.float32)
                allowed_neighbours_emb_batch = []
                for idx in batch:
                    allowed_neighbours_emb_batch.append(allowed_neighbours_emb[idx])
                allowed_neighbours_emb_batch = T.tensor(allowed_neighbours_emb_batch, dtype=torch.float32)

                old_probs = T.tensor(old_probabilities_memory[batch], dtype=T.float32)
                actions = T.tensor(actions_memory[batch])

                predicted_next_states = self._actorPredict(states)
                critic_values = self._criticPredict(states)

                distances_to_neighbours = self.distance_function(predicted_next_states, allowed_neighbours_emb_batch)
                neighbours_distributions_batch = Categorical(1 / (distances_to_neighbours + self.eps))
                new_probs = neighbours_distributions_batch.log_prob(actions)

                prob_ratio = new_probs.exp() / old_probs.exp()

                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(
                    prob_ratio,
                    1 - self.policy_ratio_clip,
                    1 + self.policy_ratio_clip
                ) * advantage[batch]

                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()
                returns = advantage[batch] + values[batch]

                critic_loss = (returns - critic_values) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + self.critic_loss_coef * critic_loss

                total_loss.backward()
                print(f'Loss: {total_loss.detach().item() / len(batch)}')
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()


# Works only with batches
def euclidean_distance(existing_state: T.Tensor, allowed_neighbours: T.Tensor):
    """Returns batch of distances from existing_state to each allowed_neighbours"""
    unsqueezed_state = existing_state.unsqueeze(1)
    return torch.sum(torch.sub(allowed_neighbours, unsqueezed_state) ** 2, dim=2)


def linear_distance(existing_state: T.Tensor, predicted_state: T.Tensor):
    pass


def cosine_distance(existing_state, predicted_state):
    pass


# TODO add another component for loss (see message history)


class PPORouterEmbNetwork(NetworkRewardAgent, PPORouterEmb):
    def registerResentPkg(self, pkg: Package, Q_estimate: float, action, data, **kwargs) -> RewardMsg:
        # Ignore Q-estimate here

        # data = state
        # action = to

        state = kwargs['state'],
        action_idx = kwargs['action_idx'],
        prob = kwargs['prob'],
        value = kwargs['value'],
        allowed_neighbours = kwargs['allowed_neighbours']

        reward_data = self._getRewardData(pkg, data)

        # self._pending_pkgs[pkg.id] = (action, reward_data, data)
        self._pending_pkgs[pkg.id] = (state, action_idx, prob, value, allowed_neighbours, action, reward_data)
        # Igor Buzhinsky's hack to suppress a no-key exception in receiveReward
        # self._last_tuple = action, reward_data, data

        return self._mkReward(pkg, Q_estimate, reward_data)

    def receiveReward(self, msg: RewardMsg):
        # try:
            # action, old_reward_data, saved_data = self._pending_pkgs.pop(msg.pkg.id)
        saved_state, action_idx, prob, value, allowed_neighbours, action, old_reward_data = self._pending_pkgs.pop(msg.pkg.id)
        # except KeyError:
        #     self.log(f'not our package: {msg.pkg}, path:\n  {msg.pkg.node_path}\n', force=True)
            # Igor Buzhinsky's hack to suppress a no-key exception in receiveReward
            # action, old_reward_data, saved_data = self._last_tuple
        reward = self._computeReward(msg, old_reward_data)
        # return action, reward, saved_data
        return saved_state, action_idx, prob, value, allowed_neighbours, action, reward


class PPORouterEmbConveyor(LSConveyorMixin, ConveyorRewardAgent, PPORouterEmb):
    def registerResentPkg(self, pkg: Package, Q_estimate: float, action, data, **kwargs) -> RewardMsg:
        state = kwargs['state'],
        action_idx = kwargs['action_idx'],
        prob = kwargs['prob'],
        value = kwargs['value'],
        allowed_neighbours = kwargs['allowed_neighbours']

        reward_data = self._getRewardData(pkg, data)

        # self._pending_pkgs[pkg.id] = (action, reward_data, data)
        self._pending_pkgs[pkg.id] = (state, action_idx, prob, value, allowed_neighbours, action, reward_data)
        # Igor Buzhinsky's hack to suppress a no-key exception in receiveReward
        # self._last_tuple = action, reward_data, data

        return self._mkReward(pkg, Q_estimate, reward_data)

    def receiveReward(self, msg: RewardMsg):
        # try:
        # action, old_reward_data, saved_data = self._pending_pkgs.pop(msg.pkg.id)
        saved_state, action_idx, prob, value, allowed_neighbours, action, old_reward_data = \
            self._pending_pkgs.pop(msg.pkg.id)
        # except KeyError:
        #     self.log(f'not our package: {msg.pkg}, path:\n  {msg.pkg.node_path}\n', force=True)
        # Igor Buzhinsky's hack to suppress a no-key exception in receiveReward
        # action, old_reward_data, saved_data = self._last_tuple
        reward = self._computeReward(msg, old_reward_data)
        # return action, reward, saved_data
        return saved_state, action_idx, prob, value, allowed_neighbours, action, reward

    def _computeReward(self, msg: ConveyorRewardMsg, old_reward_data):
        time_sent, _ = old_reward_data
        time_processed, energy_gap = msg.reward_data
        time_gap = time_processed - time_sent

        # self.log('time gap: {}, nrg gap: {}'.format(time_gap, energy_gap), True)
        # return time_gap + self._e_weight * energy_gap
        return msg.Q_estimate + time_gap + self._e_weight * energy_gap


get_distance_function = {
    'euclid': euclidean_distance,
    'linear': linear_distance,
    'cosine': cosine_distance
}
