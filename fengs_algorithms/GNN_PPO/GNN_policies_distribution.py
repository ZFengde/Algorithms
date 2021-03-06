# best GNN so far
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
from torch.distributions import Normal
import math

class DiagGaussianDistribution():
    """
    Gaussian distribution with diagonal covariance matrix, for continuous actions.

    :param action_dim:  Dimension of the action space.
    """

    def __init__(self, action_dim: int):
        super(DiagGaussianDistribution, self).__init__()
        self.action_dim = action_dim
        self.mean_actions = None
        self.log_std = None

    def proba_distribution_net(self, latent_dim: int, log_std_init: float):
        """
        Create the layers and parameter that represent the distribution:
        one output will be the mean of the Gaussian, the other parameter will be the
        standard deviation (log std in fact to allow negative values)

        :param latent_dim: Dimension of the last layer of the policy (before the action layer)
        :param log_std_init: Initial value for the log standard deviation
        :return:
        """
        mean_actions = nn.Linear(latent_dim, self.action_dim)
        log_std = nn.Parameter(torch.ones(self.action_dim) * log_std_init, requires_grad=True)
        return mean_actions, log_std

    def proba_distribution(self, mean_actions, log_std) -> "DiagGaussianDistribution":
        """
        Create the distribution given its parameters (mean, std)

        :param mean_actions:
        :param log_std:
        :return:
        """
        action_std = torch.ones_like(mean_actions) * log_std.exp()
        self.distribution = Normal(mean_actions, action_std)
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Get the log probabilities of actions according to the distribution.
        Note that you must first call the ``proba_distribution()`` method.

        :param actions:
        :return:
        """
        log_prob = self.distribution.log_prob(actions)
        return self.sum_independent_dims(log_prob)

    def entropy(self) -> torch.Tensor:
        return self.sum_independent_dims(self.distribution.entropy())

    def sample(self):
        # Reparametrization trick to pass gradients
        return self.distribution.rsample()

    def mode(self):
        return self.distribution.mean

    def actions_from_params(self, mean_actions, log_std, deterministic):
        # Update the proba distribution
        self.proba_distribution(mean_actions, log_std)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, mean_actions, log_std):
        """
        Compute the log probability of taking an action
        given the distribution parameters.

        :param mean_actions:
        :param log_std:
        :return:
        """
        actions = self.actions_from_params(mean_actions, log_std)
        log_prob = self.log_prob(actions)
        return actions, log_prob

    def sum_independent_dims(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Continuous actions are usually considered to be independent,
        so we can sum components of the ``log_prob`` or the entropy.

        :param tensor: shape: (n_batch, n_actions) or (n_batch,)
        :return: shape: (n_batch,)
        """
        if len(tensor.shape) > 1:
            tensor = tensor.sum(dim=1)
        else:
            tensor = tensor.sum()
        return tensor

    def get_actions(self, deterministic = False):
        if deterministic:
            return self.mode()
        return self.sample()

class GNN_Layer(nn.Module):
    def __init__(
                self,
                in_feat,
                out_feat,
                graph,
            ):      
        super(GNN_Layer, self).__init__()
        self.g = graph
        self.in_feat = in_feat
        
        self.loop_weight =nn.Parameter(torch.Tensor(graph.num_nodes(), in_feat, out_feat))
        self.W = nn.Parameter(torch.Tensor(graph.num_edges(), in_feat, out_feat))
        self.m_bias = nn.Parameter(torch.Tensor(graph.num_edges(), 1, out_feat))
        self.h_bias = nn.Parameter(torch.Tensor(graph.num_nodes(), 1, out_feat))
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.uniform_(self.W, -1/math.sqrt(self.in_feat), 1/math.sqrt(self.in_feat))
        nn.init.zeros_(self.m_bias)
        nn.init.zeros_(self.h_bias)

    def message(self, edges):
        if edges.src['h'].dim() == 2:
            x = edges.src['h'].view(-1, 1, self.in_feat)
            message = torch.bmm(x, self.W) + self.m_bias
        elif edges.src['h'].dim() == 3:
            message = torch.bmm(edges.src['h'], self.W) + self.m_bias
        return {'m' : message}

    def forward(self, feat):
        with self.g.local_scope():
            self.g.srcdata['h'] = feat

            # self-loop
            if self.g.srcdata['h'].dim() == 2:
                x = self.g.srcdata['h'].view(-1, 1, self.in_feat)
                loop = torch.bmm(x,  self.loop_weight)
            elif self.g.srcdata['h'].dim() == 3:
                loop = torch.bmm(self.g.srcdata['h'], self.loop_weight)
            self.g.update_all(self.message, fn.sum('m', 'h'))
            h = self.g.dstdata['h'] + self.h_bias + loop
            return h.squeeze()

class GNN(nn.Module):
    def __init__(
                self,
                in_feat,
                out_feat,
                graph,
            ):
        super(GNN, self).__init__()
        self.layer1 = GNN_Layer(in_feat, 8, graph)
        self.layer2 = GNN_Layer(8, out_feat, graph)

    def forward(self, features):
        x = torch.tanh(self.layer1(features))
        x = self.layer2(x)
        return x

class Trial_GNN_ActorCriticPolicy(nn.Module):
    def __init__(
        self, 
        node_input_dim: int = 6,
        node_output_dim: int = 8,
        actor_output_dim : int = 2, 
        device = None,
        log_std_init = 0.0, # according to the StateDependentNoiseDistribution class from baseline3
    ):
        super(Trial_GNN_ActorCriticPolicy, self).__init__()
        self.device = device
        src_ids = torch.tensor([0, 0, 0, 1, 2, 1])
        dst_ids = torch.tensor([1, 2, 3, 2, 3, 3])
        self.g = dgl.graph((src_ids, dst_ids)).to(device)
        self.log_std_init = log_std_init

        self.gnn = GNN(node_input_dim, node_output_dim, self.g)
        self.common_layer = nn.Linear(32, 64)
        self.actor_latent_layer = nn.Linear(64, 64)
        self.action_dist = DiagGaussianDistribution(action_dim=actor_output_dim)
        self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=64, log_std_init=self.log_std_init)
        self.critic_latent_layer = nn.Linear(64, 64)
        self.value_net = nn.Linear(64, 1)

    def forward(self, obs, t_1_info, t_2_info, mode='sample'):
        # actor network
        features = self.batch_gnn_process(obs, t_1_info, t_2_info)
        shared_latent = torch.tanh(self.common_layer(features))
        latent_pi = torch.tanh(self.actor_latent_layer(shared_latent))
        distributions = self._get_action_dist_from_latent(latent_pi)
        actions = self.get_actions(distributions, mode=mode)
        log_probs = distributions.log_prob(actions)
        
        # critic network
        latent_vf = torch.tanh(self.critic_latent_layer(shared_latent))
        values = self.value_net(latent_vf)

        return actions, values, log_probs
    
    def _get_action_dist_from_latent(self, latent_pi):
        mean_actions = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(mean_actions, self.log_std)

    def get_actions(self, distribution, mode):
        if mode == 'sample':
            return distribution.sample()
        elif mode =='deterministic':
            return distribution.mode()

    def predict_values(self, obs, t_1_info, t_2_info):
        if obs.dim() == 2:
            features = self.batch_gnn_process(obs, t_1_info, t_2_info)
        else:
            features = self.gnn_process(obs, t_1_info, t_2_info)
        shared_latent = torch.tanh(self.common_layer(features))
        latent_vf = torch.tanh(self.critic_latent_layer(shared_latent))
        values = self.value_net(latent_vf)

        return values

    def evaluate_actions(self, obs, actions, t_1_info, t_2_info):
        if obs.dim() == 2:
            features = self.batch_gnn_process(obs, t_1_info, t_2_info)
        else:
            features = self.gnn_process(obs, t_1_info, t_2_info)
        shared_latent = torch.tanh(self.common_layer(features))
        latent_pi = torch.tanh(self.actor_latent_layer(shared_latent))
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)

        latent_vf = torch.tanh(self.critic_latent_layer(shared_latent))
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

    def predict(self, obs, t_1_info, t_2_info):
        # actor network
        features = self.gnn_process(obs, t_1_info, t_2_info)
        shared_latent = torch.tanh(self.common_layer(features))
        latent_pi = torch.tanh(self.actor_latent_layer(shared_latent))
        distributions = self._get_action_dist_from_latent(latent_pi)
        actions = self.get_actions(distributions, mode='deterministic')
        return actions

    def batch_gnn_process(self, obss, t_1_infos, t_2_infos):
        # For test
        target_poss = obss[:,6:]
        target_infos = torch.cat((target_poss,torch.zeros_like(target_poss),torch.zeros_like(target_poss)),dim=1)
        nodes_infos = torch.stack((target_infos, t_1_infos, t_2_infos, obss[:, 0: 6]),dim=1) # 6, 4, 6
        nodes_infos = torch.transpose(nodes_infos, 0, 1) # 4, 6, 6, nodes, batch, info
        graph_output = torch.tanh(self.gnn(nodes_infos)) # 4, 6, 2
        graph_output = torch.transpose(graph_output, 0, 1) # 6, 4, 2
        features = torch.flatten(graph_output, start_dim=1) # 6, 8
        return features     

    def gnn_process(self, obs, t_1_info, t_2_info):
        target_pos = obs[6:]
        target_info = torch.cat((target_pos,torch.zeros_like(target_pos),torch.zeros_like(target_pos)))
        node_info = torch.cat((target_info, t_1_info.squeeze(), t_2_info.squeeze(), obs[0: 6])).view(4, 6) # 4, 6
        features = torch.tanh(self.gnn(node_info)).flatten() # 4, 2
        return features     