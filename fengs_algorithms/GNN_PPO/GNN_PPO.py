import datetime
import os
import time
import torch
import numpy as np
from torch.optim import Adam
import torch
import torch.nn.functional as F
from fengs_algorithms.GNN_PPO.policies_distribution import ActorCriticPolicy
from fengs_algorithms.common.buffer import RolloutBuffer
from fengs_algorithms.common.utils import obs_as_tensor, Logger

class GNN_PPO():
    def __init__(
        self,
        env, 
        policy = ActorCriticPolicy, 
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        n_rollout_steps: int = 2048,
        n_epochs: int = 10,
        clip_range: float = 0.2,
        ent_coef: float= 0.0,
        ac_lr: float = 3e-4,
        gnn_lr: float = 0.001,
        vf_coef: float = 0.5,
        batch_size: int = 64,
        max_grad_norm: float = 0.5,
        experiment_name = 'PPO',
        buffer_cls = RolloutBuffer,
        logger = Logger,
        save_model_name = 'PPO',
        parallel = True
    ):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # initialise hyperparameters
        self.env = env
        self.gamma = gamma
        self.n_rollout_steps = n_rollout_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.ac_lr = ac_lr
        self.gnn_lr = gnn_lr
        self.vf_coef = vf_coef
        self.gae_lambda = gae_lambda
        self.experiment_name = experiment_name
        self.max_grad_norm = max_grad_norm
        if parallel:
            self.n_envs = self.env.num_envs
        else:
            self.n_envs = 1
        self.save_model_name = save_model_name
        self.time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.rollout_buffer = buffer_cls(
            self.n_rollout_steps,
            self.env.observation_space,
            self.env.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        self.policy = policy(
            input_dim=self.env.observation_space.shape[0], 
            actor_output_dim=2,).to(self.device)

        self.logger = logger(experiment_name, self.time)
        self.num_timesteps = 0
        self._n_updates = 0 
        self.policy_optim = Adam(self.policy.parameters(), lr=self.ac_lr, eps=1e-5)

        self.episode_reward_buffer = np.zeros((self.n_envs,))
        self.episode_length_buffer = np.zeros((self.n_envs,))
        self._last_obs = None
        self._last_episode_starts = np.ones((self.n_envs,), dtype=bool)

    def collect_rollouts(self):
        # log episode information within rollout
        rollout_ep_rewards = []
        rollout_ep_len = []
        if self._last_obs is None:
            self._last_obs = self.env.reset()

        n_steps = 0
        self.rollout_buffer.reset()

        while n_steps < self.n_rollout_steps:
            with torch.no_grad():
                # simply convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()
            
            clipped_actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)
            new_obs, rewards, dones, infos = self.env.step(clipped_actions)
            
            # so num_timesteps count all environment altogether
            self.num_timesteps += self.n_envs
            n_steps += 1

            self.episode_length_buffer += 1
            self.episode_reward_buffer += rewards
            # in every step, check all envs if there is done
            # if so, pass the information to the buffer
            
            for idx, done in enumerate(dones):
                if done:
                    # log episode information within rollout 
                    # and initialise the corresponding env_index info
                    rollout_ep_rewards.append(self.episode_reward_buffer[idx])
                    rollout_ep_len.append(self.episode_length_buffer[idx])
                    self.episode_length_buffer[idx] = 0
                    self.episode_reward_buffer[idx] = 0
                    if (infos[idx].get("terminal_observation") is not None 
                            # this means if "TimeLimit.truncated" doesn't have value, then output False
                            # only when time out of timelimit, continue
                            and infos[idx].get("TimeLimit.truncated", False)):
                        # the reason why don't directly use last obs is that in _worker of 
                        # parallel vectorized env, if done,... = step(aciton), what return is obs = reset()
                        # and the real last obs is store at (infos[idx]["terminal_observation"])
                        terminal_obs = obs_as_tensor(infos[idx]["terminal_observation"], self.device)
                        with torch.no_grad():
                            terminal_value = self.policy.predict_values(terminal_obs)
                        rewards[idx] += self.gamma * terminal_value

            # done reset for specific environment is finish inside subproc_vec_env
            self.rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs)
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with torch.no_grad():
            # Compute values for the last timestep, for handle truncation
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        # for compute the returns and advantages for whole buffer
        # we need to konw the values of last step of the buffer and the corresponding done condition
        self.rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        if len(rollout_ep_rewards) > 0:
            ep_rew_mean = np.mean(rollout_ep_rewards)
            ep_len_mean = np.mean(rollout_ep_len)
            self.logger.record("rollout/ep_num_rollout", len(rollout_ep_rewards))
            self.logger.record("rollout/ep_rew_mean", ep_rew_mean)
            self.logger.record("rollout/ep_len_mean", ep_len_mean)
            self.logger.to_tensorboard(data=ep_rew_mean, time_count=self.num_timesteps)
        self.logger.record('rollout/timesteps_so_far', self.num_timesteps)

        return True
    
    def train(self):
        for _ in range(self.n_epochs):
            # here generate random small batch from rollout buffer
            # and do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.sample(self.batch_size):
                actions = rollout_data.actions

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()

                # calculate and normalise the advantages
                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                surr_loss_1 = advantages * ratio
                surr_loss_2 = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -torch.min(surr_loss_1, surr_loss_2).mean()

                values_pred = values.squeeze()
                # Value loss using the TD(gae_lambda) target 64, 16
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                accumulative_loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # TODO, GNN backprobagate here
                self.policy_optim.zero_grad()
                accumulative_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy_optim.step()
                
        self._n_updates += self.n_epochs
        # logger
        self.logger.record("train/policy_loss", policy_loss.item())
        self.logger.record("train/value_loss", value_loss.item())
        self.logger.record("train/loss", accumulative_loss.item())
        self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates)

        # save model
        self.save()

    def learn(self, total_timesteps):
        iteration = 0
        while self.num_timesteps < total_timesteps:
            self.collect_rollouts()
            self.train()
            iteration += 1
            self.logger.record("time/iterations", iteration)
            self.logger.write_out()
        return self

    def save(self):
        models_dir = os.path.join(f'{self.experiment_name}', 'models', f'{self.time}')
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        torch.save(self.policy.state_dict(), f'{models_dir}/{self.num_timesteps}_{self.save_model_name}')

    def load(self, path):
        self.policy.load_state_dict(torch.load(path))

    def test(self, test_episode):

        self.rollout_buffer.reset()

        for episode_count in range(test_episode):
            ep_reward = 0
            ep_len = 0
            obs = self.env.reset()
            while True:
                with torch.no_grad():
                    # simply convert to pytorch tensor or to TensorDict
                    obs_tensor = obs_as_tensor(obs, self.device)
                    action = self.policy.predict(obs_tensor)
                action = action.cpu().numpy()

                #this can cause suboptimal action
                action = np.clip(action, self.env.action_space.low, self.env.action_space.high)

                obs, reward, done, _ = self.env.step(action)
                if self.env.use_gui == True:
                    time.sleep(1./240.)
                ep_reward += reward
                ep_len += 1
                if done:
                    self.logger.record('episode_count', episode_count)
                    self.logger.record('test/ep_len', ep_len)
                    self.logger.record('test/ep_reward', ep_reward)
                    self.logger.write_out()
                    break
        return True