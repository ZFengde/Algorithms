import os
import time
import datetime
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from fengs_algorithms.GCN_PPO.GCN_policies_distribution import GCN_ActorCriticPolicy
from fengs_algorithms.common.buffer import Temp_RolloutBuffer
from fengs_algorithms.common.utils import obs_as_tensor, Logger

class GCN_PPO():
    def __init__(
        self,
        env, 
        policy = GCN_ActorCriticPolicy, 
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        n_rollout_steps: int = 2048,
        n_epochs: int = 10,
        clip_range: float = 0.2,
        ent_coef: float= 0.0,
        ac_lr: float = 3e-4,
        vf_coef: float = 0.5,
        batch_size: int = 64,
        max_grad_norm: float = 0.5,
        experiment_name = 'GCN_PPO',
        buffer_cls = Temp_RolloutBuffer,
        logger = Logger,
        save_model_name = 'GCN_PPO',
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
            node_input_dim=2, 
            node_output_dim=1, 
            actor_output_dim=2,
            device=self.device).to(self.device)

        self.logger = logger(experiment_name, self.time)
        self.num_timesteps = 0
        self._n_updates = 0 
        
        # self.policy_optim = Adam([
        #                         {'params': self.policy.gnn.parameters(), 'lr': 1e-3},
        #                         ], lr=3e-4, eps=1e-5)
        self.policy_optim = Adam(self.policy.parameters(), lr=self.ac_lr, eps=1e-5)

        self.episode_reward_buffer = np.zeros((self.n_envs,))
        self.episode_length_buffer = np.zeros((self.n_envs,))
        self.t_1_info = np.zeros((self.n_envs, 2))
        self.t_2_info = np.zeros((self.n_envs, 2))
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
        ep_num_success = 0

        while n_steps < self.n_rollout_steps:
            with torch.no_grad():
                temp_1 = obs_as_tensor(self.t_1_info, self.device)
                temp_2 = obs_as_tensor(self.t_2_info, self.device)
                # simply convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor, temp_1, temp_2)
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

            self.t_2_info = self.t_1_info
            self.t_1_info = new_obs[:, 0: 2]
            
            for idx, done in enumerate(dones):
                if done:
                    rollout_ep_rewards.append(self.episode_reward_buffer[idx])
                    rollout_ep_len.append(self.episode_length_buffer[idx])
                    self.episode_length_buffer[idx] = 0
                    self.episode_reward_buffer[idx] = 0
                    self.t_1_info[idx] = np.zeros((1, 2))
                    self.t_2_info[idx] = np.zeros((1, 2))
                    if infos[idx].get('Success') == 'Yes':
                        ep_num_success += 1
                    if (infos[idx].get("terminal_observation") is not None 
                            and infos[idx].get("TimeLimit.truncated", False)):
                        terminal_obs = obs_as_tensor(infos[idx]["terminal_observation"], self.device)
                        with torch.no_grad():
                            terminal_value = self.policy.predict_values(terminal_obs, temp_1[idx], temp_2[idx])
                        rewards[idx] += self.gamma * terminal_value

            self.rollout_buffer.add(
                                    self._last_obs, 
                                    actions, rewards, 
                                    self._last_episode_starts, 
                                    values, 
                                    log_probs,
                                    temp_1,
                                    temp_2)
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with torch.no_grad():
            # Compute values for the last timestep, for handle truncation
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device), temp_1, temp_2)

        # for compute the returns and advantages for whole buffer
        # we need to konw the values of last step of the buffer and the corresponding done condition
        self.rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        ep_num_rollout = len(rollout_ep_rewards)
        success_rate = ep_num_success/ep_num_rollout
        if len(rollout_ep_rewards) > 0:
            self.logger.record("rollout/ep_rew_mean", np.mean(rollout_ep_rewards))
            self.logger.record("rollout/success_rate", success_rate)
            self.logger.record("rollout/ep_num_rollout", ep_num_rollout)
            self.logger.record("rollout/ep_num_success", ep_num_success)
            self.logger.record("rollout/ep_len_mean", np.mean(rollout_ep_len))
            self.logger.to_tensorboard(name='Episode_reward_mean', data=np.mean(rollout_ep_rewards), time_count=self.num_timesteps)
            self.logger.to_tensorboard(name='Success_rate', data=success_rate, time_count=self.num_timesteps)
            self.logger.close()
        self.logger.record('rollout/timesteps_so_far', self.num_timesteps)

        return True
    
    def train(self):
        for i in range(self.n_epochs):
            print(i)
            # here generate random small batch from rollout buffer
            # and do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.sample(self.batch_size):
                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, 
                                                                        rollout_data.actions, 
                                                                        rollout_data.t_1_infos, 
                                                                        rollout_data.t_2_infos)
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
        models_dir = os.path.join(f'{self.experiment_name}_Record', 'models', f'{self.time}')
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            print(f'logging to {models_dir}')
        torch.save(self.policy.state_dict(), f'{models_dir}/{self.num_timesteps}')

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
                    # TODO, need to be added with temporal information here, not urgent 
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