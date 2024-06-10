import os
import time
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from IPython.display import clear_output
import pandas as pd
import numpy as np

from a_actor_critic import CnnEncoder, ContinuousActor, Critic, init_weights, Memory
from c_ppo_utils import get_gae, trajectories_data_generator
        
import wandb
from datetime import datetime
from collections import deque
from pytz import timezone
from copy import deepcopy

class PPOAgent(object):
    """PPOAgent.
    Parameters:
        device: cpu or gpu acelator.
        make_env: factory that produce environment.
        continuous: True of environments with continuous action space.
        obs_dim: dimension od observaion.
        act_dim: dimension of action.
        gamma: coef for discount factor.
        lamda: coef for general adversial estimator (GAE).
        entropy_coef: coef of weighting entropy in objective loss.
        epsilon: clipping range for actor objective loss.
        actor_lr: learnig rate for actor optimizer.
        critic_lr: learnig rate for critic optimizer.
        value_range: clipping range for critic objective loss.
        rollout_len: num t-steps per one rollout.
        total_rollouts: num rollouts.
        num_epochs: num weights updation iteration for one policy update.
        batch_size: data batch size for weights updating
        actor: model for predction action.
        critic: model for prediction state values.
        plot_interval: interval for plotting train history.
        solved_reward: desired reward.
        plot_interval: plot history log every plot_interval rollouts.
        path2save_train_history: path to save training history logs.
        """
    def __init__(self, make_env, args):
        """
        Initialization.
        """
        self.device = args.device
        self.env_name = args.env_name
        self.env = make_env(args.env_name, config=args)
        print("device:", self.device)
        # self.env = make_env(args.env_name, args)
        self.start_time = datetime.now(timezone('Asia/Seoul')).strftime("%y%m%d%H%M")

        # wandb log name
        self.run_number = args.run_number
        self.save_path_with_time = self.env_name[9:] + '_' + str(self.run_number) + '_' + str(self.start_time[2:])

        # coeffs
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.entropy_coef = args.entropy_coef
        self.epsilon = args.epsilon
        self.value_range = args.value_range
        
        # other hyperparameters
        self.rollout_len = args.rollout_len
        self.total_rollouts = args.total_rollouts
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size

        # agent nets
        self.obs_dim = args.obs_dim
        self.encoder = CnnEncoder().apply(init_weights).to(self.device)
        self.actor = ContinuousActor(self.encoder, self.device).apply(init_weights).to(self.device)
        self.critic = Critic(self.encoder, self.device).apply(init_weights).to(self.device)

        # agent nets optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr)

        # memory of trajectory (s, a, r ...)
        self.memory = Memory()

        # memory of the train history
        self.actor_loss_history = []
        self.critic_loss_history = []
        self.scores = []

        self.best_score = -10
        self.best_score_save = False

        self.is_evaluate = args.is_evaluate
        self.solved_reward = args.solved_reward
        self.plot_interval = args.plot_interval
        self.print_episode_interval = args.print_episode_interval
        self.path2save_train_history = args.path2save_train_history
        self.flag_solved = False
        self.period_save_model = args.period_save_model
        self.step_save = False

        # load model
        if args.load_model:
            self.actor.load_state_dict(torch.load(args.load_model_actor_path, map_location=self.device))
            self.critic.load_state_dict(torch.load(args.load_model_critic_path, map_location=self.device))
            self.encoder.load_state_dict(torch.load(args.load_model_encoder_path, map_location=self.device))

            print("COMPLETE MODEL LOAD!!!!")

        # wandb
        self.wandb_use = args.wandb_use
        self.train_flag = False
        self.time_step = 0
        self.num_episode = 0
        self.num_train = 0
        self.episode_reward_list = deque(maxlen=30)
        self.actor_loss_list = deque(maxlen=30)
        self.entropy_loss_list = deque(maxlen=30)
        self.critic_loss_list = deque(maxlen=30)
        self.ratio_list = deque(maxlen=30)

        if self.wandb_use and not self.is_evaluate:
            now_time = datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S')
            wandb_name = f"{args.env_name}_{now_time}"
            self.wandb = wandb.init(
                project=f"{args.env_name}",
                name=wandb_name,
                save_code=False
            )

    def _get_action(self, state_me: np.ndarray, state_you: np.ndarray) -> tuple:
    # def _get_action(self, state_me: np.ndarray) -> float:

        """
        Get action from actor, and if not test -  
        get state value from critic, collect elements of trajectory.
        """
        # print(f"{state = }")
        state_me = np.array(state_me)
        state_me = torch.FloatTensor(state_me).to(self.device)
        state_you = np.array(state_you)
        state_you = torch.FloatTensor(state_you).to(self.device)

        # print(f"{state.shape = }, {state.dim() = }, *************************")
        # print(f"{state = }")
        action_me, dist_me = self.actor(state_me)
        # print(action_me)
        if self.copy_on :
            action_you, dist_you = self.copied_actor(state_you)
        else:
            action_you = torch.tensor([[0.0, 0.0]]).to('cuda:0')

        if not self.is_evaluate:
            value = self.critic(state_me)
           
            # collect elements of trajectory
            self.memory.states.append(state_me)
            self.memory.actions.append(action_me)
            self.memory.log_probs.append(dist_me.log_prob(action_me))
            self.memory.values.append(value)

        #return list(action_me.detach().cpu().numpy()).pop()
        return list(action_me.detach().cpu().numpy()).pop(), list(action_you.detach().cpu().numpy()).pop()
            
            
    def _step(self, action_me: float, action_you: float):
        """
        Make action in enviroment chosen by current policy,
        if not evaluate - collect elements of trajectory.
        """
        # print(action_me)
        #next_state_me, reward, terminated, truncated, _ = self.env.step(action_me) # 🔥 next_state_you를 만들어야 함
        next_state_me, reward, terminated, truncated, _, next_state_you = self.env.step(action_me, action_you)

        if any([terminated, truncated]):
            done = True
        else:
            done = False

        # add fake dim to match dimension with batch size
        next_state_me = np.reshape(next_state_me, (1, 4, 40, 40)).astype(np.float64)
        next_state_you = np.reshape(next_state_you, (1, 4, 40, 40)).astype(np.float64)
        reward = np.reshape(reward, (1, -1)).astype(np.float64)
        done = np.reshape(done, (1, -1))

        if not self.is_evaluate:
            # convert np.ndarray return from enviroment to torch tensor. 
            # collect elements of trajectory.
            reward = np.array(reward)
            done_memory = np.array(1 - done)

            self.memory.rewards.append(torch.FloatTensor(reward).to(self.device))
            self.memory.is_terminals.append(torch.FloatTensor(done_memory).to(self.device))

        # return next_state_me, reward, done
        return next_state_me, reward, done, next_state_you

    def train(self):
        """
        Interaction process in enviroment for collect trajectory,
        train process by agent nets after each rollout.
        """
        total_train_start_time = time.time()

        score = 0
        # state_me, _ = self.env.reset()
        state_me, _, state_you = self.env.reset()
        state_me = np.asarray(state_me)
        state_you = np.asarray(state_you)

        self.num_episode = 0
        self.time_step = 0
        episode_reward = 0
        print_episode_flag = False

        for step_ in range(self.total_rollouts):
            for _ in range(self.rollout_len):
                # action_me = self._get_action(state_me)
                action_me, action_you = self._get_action(state_me, state_you)
                # next_state_me, reward, done = self._step(action_me)
                next_state_me, reward, done, next_state_you = self._step(action_me, action_you)

                state_me = next_state_me
                state_you = next_state_you

                score += reward[0][0]
                episode_reward += reward[0][0]

                # print(f"{test.shape = } $@!#$%$#*&&**()*)(*(*)()*")

                if done[0][0]:
                    self.scores.append(score)
                    score = 0
                    # state_me, _ = self.env.reset()
                    state_me, _, state_you = self.env.reset()
                    self.num_episode += 1
                    self.episode_reward_list.append(episode_reward)
                    episode_reward = 0
                    print_episode_flag = True

                self.time_step += 1

                total_training_time = time.time() - total_train_start_time
                total_training_time = time.strftime('%H:%M:%S', time.gmtime(total_training_time))
                if self.num_episode % self.print_episode_interval == 0 and print_episode_flag:
                    print(
                        "[Episode {:3,}, Steps {:6,}]".format(self.num_episode, self.time_step),
                        "Episode Reward: {:>9.3f},".format(np.mean(np.asarray(self.episode_reward_list))),
                        "Elapsed Time: {}".format(total_training_time)
                    )
                    print_episode_flag = False

                if len(self.episode_reward_list) > 0:
                    if (sum(self.episode_reward_list) / len(self.episode_reward_list)) > self.best_model:
                        self.best_model = (sum(self.episode_reward_list) / len(self.episode_reward_list))
                        print("📌 Best model saved!!")
                        self.best_model_save = True
                        self._save_train_history()

            if step_ % self.plot_interval == 0 and self.wandb_use and self.train_flag and not self.is_evaluate:
                # self._plot_train_history()
                self.log_wandb()

            if self.time_step % self.period_save_model == 0:
                self.step_save = True
                self._save_train_history()

            # if we have achieved the desired score - stop the process.
            if self.solved_reward is not None:
                if np.mean(self.scores[-20:]) > self.solved_reward:
                    self.flag_solved = True
                    print(f"It's solved! 20 episode reward mean = {np.mean(self.scores[-20:])}")
                    break
                else:
                    if np.mean(self.scores[-20:]) > self.best_score:
                        self.best_score = np.mean(self.scores[-10:])
                        print("🔥 Best score saved!!")
                        self.best_score_save = True
                        self._save_train_history()


            next_state_me = np.array(next_state_me)
            value = self.critic(torch.FloatTensor(next_state_me).to(self.device))
            self.memory.values.append(value)
            # update policy
            self._update_weights()
            self.num_train += 1

        self._save_train_history()
        self.env.close()

    def _update_weights(self):

        returns = get_gae(
            self.memory.rewards,
            self.memory.values,
            self.memory.is_terminals,
            self.gamma,
            self.lamda,
        )
        actor_losses, critic_losses = [], []
        actor_wo_entropy_losses, entropy_losses = [], []
        ratio_list = []

        # flattening a list of torch.tensors into vectors
        states = torch.cat(self.memory.states).view(-1, 4, 40, 40)
        actions = torch.cat(self.memory.actions)
        returns = torch.cat(returns).detach()
        log_probs = torch.cat(self.memory.log_probs).detach()
        values = torch.cat(self.memory.values).detach()
        advantages = returns - values[:-1]

        for state, action, return_, old_log_prob, old_value, advantage in trajectories_data_generator(
            states=states,
            actions=actions,
            returns=returns,
            log_probs=log_probs,
            values=values,
            advantages=advantages,
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            ):

            # compute ratio (pi_theta / pi_theta__old)
            _, dist = self.actor(state)
            cur_log_prob = dist.log_prob(action)
            ratio = torch.exp(cur_log_prob - old_log_prob)

            ratio_list.append(ratio.mean().item())

            # compute entropy
            entropy = dist.entropy().mean()

            # compute actor loss
            loss =  advantage * ratio
            clipped_loss = (
                torch.clamp(ratio, 1. - self.epsilon, 1. + self.epsilon)
                 * advantage
                )
            actor_loss = (
                -torch.mean(torch.min(loss, clipped_loss))
                - entropy * self.entropy_coef)

            actor_loss_wo_entropy = -torch.mean(torch.min(loss, clipped_loss))
            entropy_loss = -torch.mean(entropy * self.entropy_coef)

            # critic loss, uncoment for clipped value loss too.
            cur_value = self.critic(state)

            critic_loss = (return_ - cur_value).pow(2).mean()

            # actor optimizer step
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            actor_loss.backward()
            critic_loss.backward()

            self.actor_optimizer.step()
            self.critic_optimizer.step()


            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

            actor_wo_entropy_losses.append(actor_loss_wo_entropy.item())
            entropy_losses.append(entropy_loss.item())

            self.actor_loss_list.append(actor_loss_wo_entropy.item())
            self.entropy_loss_list.append(entropy_loss.item())
            self.critic_loss_list.append(critic_loss.item())

        if self.time_step % 100 == 0:
            self.copy_on = True
            self.copied_actor = copy.deepcopy((self.actor))
            print("🔥🔥🔥 network copied 🔥🔥🔥")

        # clean memory of trajectory
        self.memory.clear_memory()

        # write mean losses in train history logs
        # if len(actor_losses) > 0:
        actor_loss = sum(actor_losses) / len(actor_losses)
        critic_loss = sum(critic_losses) / len(critic_losses)

        actor_wo_entropy_loss = sum(actor_wo_entropy_losses) / len(actor_wo_entropy_losses)
        entropy_loss = sum(entropy_losses) / len(entropy_losses)
        ratio = sum(ratio_list) / len(ratio_list)

        self.actor_loss_history.append(actor_loss)
        self.critic_loss_history.append(critic_loss)

        self.actor_loss_list.append(actor_wo_entropy_loss)
        self.entropy_loss_list.append(entropy_loss)
        self.critic_loss_list.append(critic_loss)
        self.ratio_list.append(ratio)

        self.train_flag = True

    def _plot_train_history(self):
        data = [self.scores, self.actor_loss_history, self.critic_loss_history]
        labels = [f"score {np.mean(self.scores[-10:])}",
                  f"actor loss {np.mean(self.actor_loss_history[-10:])}", 
                  f"critic loss {np.mean(self.critic_loss_history[-10:])}",
                  ]
        clear_output(True)
        with plt.style.context("seaborn-bright"):
            fig, axes = plt.subplots(3, 1, figsize=(6, 8))
            for i, ax in enumerate(axes):
                ax.plot(data[i], c="crimson")
                ax.set_title(labels[i])

            plt.tight_layout()
            plt.show()

    def _save_train_history(self):
        """writing model weights and training logs to files."""
        data_time = datetime.now()
        if not os.path.exists(f"{self.path2save_train_history}/{self.env_name}/{self.save_path_with_time}"):
            os.makedirs(f"{self.path2save_train_history}/{self.env_name}/{self.save_path_with_time}")

        if self.best_score_save == True:
            torch.save(self.actor.state_dict(),
                       f"{self.path2save_train_history}/{self.env_name}/{self.save_path_with_time}/score_actor.pth")
            torch.save(self.critic.state_dict(),
                       f"{self.path2save_train_history}/{self.env_name}/{self.save_path_with_time}/score_critic.pth")
            torch.save(self.encoder.state_dict(),
                       f"{self.path2save_train_history}/{self.env_name}/{self.save_path_with_time}/score_encoder.pth")
            print(f"MODEL SAVE SUCCESS!!! MODEL_DIRECTORY: {self.save_path_with_time}")
            pd.DataFrame({"actor loss": self.actor_loss_history,
                          "critic loss": self.critic_loss_history}
                         ).to_csv(
                f"{self.path2save_train_history}/{self.env_name}/{self.save_path_with_time}/score_loss_logs.csv")

            pd.DataFrame(
                data=self.scores, columns=["scores"]
            ).to_csv(f"{self.path2save_train_history}/{self.env_name}/{self.save_path_with_time}/score_score_logs.csv")
            self.best_score_save = False

        if self.best_model_save == True:
            torch.save(self.actor.state_dict(),
                       f"{self.path2save_train_history}/{self.env_name}/{self.save_path_with_time}/best_actor.pth")
            torch.save(self.critic.state_dict(),
                       f"{self.path2save_train_history}/{self.env_name}/{self.save_path_with_time}/best_critic.pth")
            torch.save(self.encoder.state_dict(),
                       f"{self.path2save_train_history}/{self.env_name}/{self.save_path_with_time}/best_encoder.pth")
            print(f"MODEL SAVE SUCCESS!!! MODEL_DIRECTORY: {self.save_path_with_time}")
            pd.DataFrame({"actor loss": self.actor_loss_history,
                          "critic loss": self.critic_loss_history}
                         ).to_csv(
                f"{self.path2save_train_history}/{self.env_name}/{self.save_path_with_time}/best_logs.csv")

            pd.DataFrame(
                data=self.scores, columns=["scores"]
            ).to_csv(f"{self.path2save_train_history}/{self.env_name}/{self.save_path_with_time}/best_score_logs.csv")
            self.best_model_save = False
        elif self.flag_solved:
            torch.save(self.actor.state_dict(),
                       f"{self.path2save_train_history}/{self.env_name}/{self.save_path_with_time}/{data_time.minute}_actor_solved.pth")
            torch.save(self.critic.state_dict(),
                       f"{self.path2save_train_history}/{self.env_name}/{self.save_path_with_time}/{data_time.minute}_critic_solved.pth")
            torch.save(self.encoder.state_dict(),
                       f"{self.path2save_train_history}/{self.env_name}/{self.save_path_with_time}/{data_time.minute}_encoder_solved.pth")
            print(f"MODEL SAVE SUCCESS!!! MODEL_DIRECTORY: {self.save_path_with_time}")
        elif self.step_save:
            torch.save(self.actor.state_dict(),
                       f"{self.path2save_train_history}/{self.env_name}/{self.save_path_with_time}/{self.time_step}_actor.pth")
            torch.save(self.critic.state_dict(),
                       f"{self.path2save_train_history}/{self.env_name}/{self.save_path_with_time}/{self.time_step}_critic.pth")
            torch.save(self.encoder.state_dict(),
                       f"{self.path2save_train_history}/{self.env_name}/{self.save_path_with_time}/{self.time_step}_encoder.pth")
            # print(f"MODEL SAVE SUCCESS!!! MODEL_DIRECTORY: {self.save_path_with_time}")
            self.step_save = False

        # pd.DataFrame({"actor loss": self.actor_loss_history,
        #               "critic loss": self.critic_loss_history}
        #              ).to_csv(f"{self.path2save_train_history}/{self.env_name}/{self.save_path_with_time}/loss_logs.csv")
        #
        # pd.DataFrame(
        #     data=self.scores, columns=["scores"]
        #     ).to_csv(f"{self.path2save_train_history}/{self.env_name}/{self.save_path_with_time}/score_logs.csv")


    def evaluate(self):
        self.is_evaluate = True

        state, _ = self.env.reset()
        done = False

        for _ in range(self.rollout_len):
            while not done:
                action = self._get_action(state)
                next_state, reward, done = self._step(action)
                state = next_state

            self.env.close()

    def load_predtrain_model(self, args):
        self.actor.load_state_dict(torch.load(args.load_model_actor_path, map_location=self.device))
        self.critic.load_state_dict(torch.load(args.load_model_critic_path, map_location=self.device))
        self.encoder.load_state_dict(torch.load(args.load_model_encoder_path, map_location=self.device))
        print("Predtrain models loaded")

    def log_wandb(self):
        log_dict = {
            "Num/train": self.num_train,
            "Num/episodes": self.num_episode,
            "Num/timesteps": self.time_step,
            "Train/episode_reward_mean": sum(self.episode_reward_list) / len(self.episode_reward_list),
            "Train/ratio": sum(self.ratio_list) / len(self.ratio_list),
            "Loss/actor_loss": sum(self.actor_loss_list) / len(self.actor_loss_list),
            "Loss/critic_loss": sum(self.critic_loss_list) / len(self.critic_loss_list),
            "Loss/entropy_loss": sum(self.entropy_loss_list) / len(self.entropy_loss_list),
        }

        wandb.log(log_dict)
