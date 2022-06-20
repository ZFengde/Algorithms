import torch.nn as nn
import torch
from torch.distributions import Normal

class StateDependentNoiseDistribution():
    """
    Distribution class for using generalized State Dependent Exploration (gSDE).
    Paper: https://arxiv.org/abs/2005.05719

    It is used to create the noise exploration matrix and
    compute the log probability of an action with that noise.

    :param action_dim: Dimension of the action space.
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,)
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this ensures bounds are satisfied.
    :param learn_features: Whether to learn features for gSDE or not.
        This will enable gradients to be backpropagated through the features
        ``latent_sde`` in the code.
    :param epsilon: small value to avoid NaN due to numerical imprecision.
    """

    def __init__(
        self,
        action_dim: int,
        use_expln: bool = False,
        epsilon: float = 1e-6,
        device = None,
        n_envs = int,
    ):
        self.device = device
        self.action_dim = action_dim
        self.latent_sde_dim = None
        self.mean_actions = None
        self.log_std = None
        self.weights_dist = None
        self.exploration_mat = None
        self.exploration_matrices = None
        self._latent_sde = None
        self.use_expln = use_expln
        self.epsilon = epsilon
        self.n_envs = n_envs

    def get_std(self, log_std):
        """
        Get the standard deviation from the learned parameter
        (log of it by default). This ensures that the std is positive.

        :param log_std:
        :return:
        """
        if self.use_expln:
            # From gSDE paper, it allows to keep variance
            # above zero and prevent it from growing too fast
            # this is a new way a piecewise funciton
            below_threshold = torch.exp(log_std) * (log_std <= 0)

            # Avoid NaN: zeros values that are below zero
            safe_log_std = log_std * (log_std > 0) + self.epsilon
            above_threshold = (torch.log1p(safe_log_std) + 1.0) * (log_std > 0)
            std = below_threshold + above_threshold
        else:
            # Use normal exponential
            std = torch.exp(log_std)

        # return a latent_sde_dim * action_dim dimension std
        return torch.ones(self.latent_sde_dim, self.action_dim).to(log_std.device) * std

    def sample_weights(self, log_std, batch_size = 1):
        """
        Sample weights for the noise exploration matrix,
        using a centered Gaussian distribution.

        :param log_std:
        :param batch_size:
        """
        std = self.get_std(log_std)
        self.weights_dist = Normal(torch.zeros_like(std), std)
        # here rsample is sample from distribution that, loc = zeros, scale = std * 1
        # which can be backpropagate
        self.exploration_mat = self.weights_dist.rsample()
        self.exploration_matrices = self.weights_dist.rsample((batch_size,))

    def proba_distribution_net(self, latent_dim, log_std_init = -2.0):
        """
        Create the layers and parameter that represent the distribution:
        one output will be the deterministic action, the other parameter will be the
        standard deviation of the distribution that control the weights of the noise matrix.

        :param latent_dim: Dimension of the last layer of the policy (before the action layer)
        :param log_std_init: Initial value for the log standard deviation
        :param latent_sde_dim: Dimension of the last layer of the features extractor
            for gSDE. By default, it is shared with the policy network.
        :return:
        """
        actions_net = nn.Linear(latent_dim, self.action_dim)
        # When we learn features for the noise, the feature dimension
        # can be different between the policy and the noise network

        self.latent_sde_dim = latent_dim
        log_std = torch.ones(self.latent_sde_dim, self.action_dim)
        log_std = nn.Parameter(log_std * log_std_init, requires_grad=True)
        # log_std is the matrix mapping latent to action epsilon

        self.sample_weights(log_std)
        # this is for generating weights_dist and exploration_mat based on log_std
        return actions_net, log_std

    def proba_distribution(self, mean_actions, log_std, latent_sde):
        """
        Create the distribution given its parameters (mean, std)

        :param mean_actions:
        :param log_std:
        :param latent_sde:
        :return:
        """
        # Stop gradient if we don't want to influence the features
        # create action distributions based on given informaiton
        self._latent_sde = latent_sde

        # dim1 = latent, dim2 = latent * aciton_dim, dim1 * dim2 = latent * action_dim
        # this is actually eplison(latent, theta_epsilon), i.e., sde epsilon
        variance = torch.mm(self._latent_sde**2, self.get_std(log_std) ** 2)
        # TODO, NaN error happens here
        # but it's impossible that variance + self.epsilon < 0
        self.distribution = Normal(mean_actions, torch.sqrt(variance + self.epsilon))
        return self

    def log_prob(self, actions):
        gaussian_actions = actions
        # log likelihood for a gaussian
        log_prob = self.distribution.log_prob(gaussian_actions)
        # Sum along action dim among multi aciton space
        log_prob = self.sum_independent_dims(log_prob)

        return log_prob

    def entropy(self):
        return self.sum_independent_dims(self.distribution.entropy())

    def sum_independent_dims(self, tensor):
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

    def sample(self):
        noise = self.get_noise(self._latent_sde)
        actions = self.distribution.mean + noise
        return actions

    def mode(self):
        actions = self.distribution.mean
        return actions

    def get_noise(self, latent_sde):
        device = latent_sde.device
        # Default case: only one exploration matrix len(self.exploration_matrices = batch_size)
        if len(latent_sde) == 1 or len(latent_sde) != len(self.exploration_matrices):
            return torch.mm(latent_sde, self.exploration_mat.to(device))

        # Use batch matrix multiplication for efficient computation
        # (batch_size, n_features) -> (batch_size, 1, n_features)
        latent_sde = latent_sde.unsqueeze(1)
        # (batch_size, 1, n_actions)
        noise = torch.bmm(latent_sde, self.exploration_matrices)
        return noise.squeeze(1)

    def actions_from_params(self, mean_actions, log_std, latent_sde, deterministic = False):
        # Update the proba distribution
        self.proba_distribution(mean_actions, log_std, latent_sde)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, mean_actions, log_std, latent_sde):
        actions = self.actions_from_params(mean_actions, log_std, latent_sde)
        log_prob = self.log_prob(actions)
        return actions, log_prob

    def get_actions(self, deterministic = False):
        if deterministic:
            return self.mode()
        return self.sample()

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
        # TODO: allow action dependent std
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

class ActorCriticPolicy(nn.Module):
    def __init__(
        self, 
        input_dim : int, 
        actor_output_dim : int, 
        log_std_init = 0.0 # according to the StateDependentNoiseDistribution class from baseline3
    ):
        super(ActorCriticPolicy, self).__init__()
        self.log_std_init = log_std_init
        latent_dim_pi = 64
        
        self.common_layer = nn.Linear(input_dim, 64)
        self.actor_latent_layer = nn.Linear(64, latent_dim_pi)
        self.critic_latent_layer = nn.Linear(64, 64)

        self.action_dist = DiagGaussianDistribution(action_dim=actor_output_dim)

        self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        self.value_net = nn.Linear(64, 1)

    def forward(self, obs, mode='sample'):
        shared_latent = torch.tanh(self.common_layer(obs))
        latent_vf = torch.tanh(self.critic_latent_layer(shared_latent))
        latent_pi = torch.tanh(self.actor_latent_layer(shared_latent))

        distributions = self._get_action_dist_from_latent(latent_pi)
        actions = self.get_actions(distributions, mode=mode)
        log_probs = distributions.log_prob(actions)
        values = self.value_net(latent_vf)

        return actions, values, log_probs
    
    def _get_action_dist_from_latent(self, latent_pi):
        mean_actions = self.action_net(latent_pi)
        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)

    def get_actions(self, distribution, mode):
        if mode == 'sample':
            return distribution.sample()
        elif mode =='deterministic':
            return distribution.mean()

    def predict_values(self, obs):
        shared_latent = torch.tanh(self.common_layer(obs))

        latent_vf = torch.tanh(self.critic_latent_layer(shared_latent))
        values = self.value_net(latent_vf)

        return values

    def evaluate_actions(self, obs, actions):
        shared_latent = torch.tanh(self.common_layer(obs))

        latent_vf = torch.tanh(self.critic_latent_layer(shared_latent))
        latent_pi = torch.tanh(self.actor_latent_layer(shared_latent))

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

    def predict(self, obs):
        shared_latent = torch.tanh(self.common_layer(obs))
        latent_pi = torch.tanh(self.actor_latent_layer(shared_latent))
        actions = self.action_net(latent_pi)

        return actions

    def reset_noise(self, n_envs: int = 1) -> None:
        """
        Sample new weights for the exploration matrix.

        :param n_envs:
        """
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), "reset_noise() is only available when using gSDE"
        self.action_dist.sample_weights(self.log_std, batch_size=n_envs)