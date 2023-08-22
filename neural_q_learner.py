import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import agent
from torch.distributions import Categorical
from random import randrange
from cooperative_craft_world import CooperativeCraftWorldState
from dqn import DQN
from transition_table import TransitionTable
from dialog import Dialog


class NeuralQLearner(agent.Agent):

    def __init__(self, name, agent_params, transition_params):

        self.agent_params = agent_params
        self.transition_params = transition_params

        self.eval_mode = agent_params["eval_mode"]
        self.agent_type = agent_params["agent_type"]
        self.log_dir = agent_params["log_dir"]
        self.saved_model_dir = agent_params["saved_model_dir"]

        self.state_size = agent_params["dqn_config"].state_size
        self.n_actions = agent_params["dqn_config"].n_actions
        self.gpu = agent_params["dqn_config"].gpu
        self.noisy_nets = agent_params["dqn_config"].noisy_nets
        self.device = torch.device("cuda" if self.gpu >= 0 else "cpu")

        self.n_step_n = agent_params["n_step_n"]
        self.max_reward = agent_params["max_reward"]
        self.min_reward = agent_params["min_reward"]
        self.exploration_style = agent_params["exploration_style"]
        self.softmax_temperature = agent_params["softmax_temperature"]
        self.ep_start = agent_params["ep_start"]
        self.ep = self.ep_start
        self.ep_end = agent_params["ep_end"]
        self.ep_endt = agent_params["ep_endt"]
        self.eval_ep = agent_params["eval_ep"]

        self.discount = agent_params["discount"]

        self.mixed_monte_carlo_proportion_start = agent_params["mixed_monte_carlo_proportion_start"]
        self.mixed_monte_carlo_proportion_endt = agent_params["mixed_monte_carlo_proportion_endt"]

        self.learn_start = agent_params["learn_start"]
        self.update_freq = agent_params["update_freq"]
        self.n_replay = agent_params["n_replay"]

        self.minibatch_size = agent_params["minibatch_size"]
        self.target_refresh_steps = agent_params["target_refresh_steps"]
        self.show_graphs = agent_params["show_graphs"]
        self.graph_save_freq = agent_params["graph_save_freq"]

        self.post_episode_return_calcs_needed = agent_params["post_episode_return_calcs_needed"]

        self.network = DQN(agent_params["dqn_config"])
        self.target_network = DQN(agent_params["dqn_config"])
        self.target_network.load_state_dict(self.network.state_dict())
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=agent_params["adam_lr"], betas=(agent_params["adam_beta1"], agent_params["adam_beta2"]), eps=agent_params["adam_eps"])

        self.numSteps = 0

        # For inserting complete episodes into the experience replay cache
        if self.post_episode_return_calcs_needed:
            self.current_episode = []
            
        self.lastState = None
        self.lastAction = None
        self.lastTerminal = False
        
        self.bestq = np.zeros((1), dtype=np.float32)

        self.transitions = TransitionTable(self.transition_params)

        self.episode_score = 0
        self.episode_score_clipped = 0
        self.moving_average_score = 0
        self.moving_average_score_clipped = 0
        self.moving_average_score_mom = 0.98
        self.moving_average_score_updates = 0

        self.q_values_plot = Dialog()
        self.score_plot = Dialog()
        self.clipped_score_plot = Dialog()

        super().__init__(name)


    def add_episode_to_cache(self):

        IDX_STATE = 0
        IDX_ACTION = 1
        IDX_EXTRINSIC_REWARD = 2
        IDX_TERMINAL = 3

        ep_length = len(self.current_episode)
        ret = np.zeros((ep_length), dtype=np.float32)
        ret_partial = np.zeros((ep_length), dtype=np.float32)

        last_n_rewards_discounted = np.zeros((self.n_step_n), dtype=np.float32)
        last_n_rewards_idx = 0

        i = ep_length - 1

        ret[i] = self.current_episode[i][IDX_EXTRINSIC_REWARD]

        last_n_rewards_discounted[last_n_rewards_idx] = self.current_episode[i][IDX_EXTRINSIC_REWARD]
        last_n_rewards_idx = (last_n_rewards_idx + 1) % self.n_step_n

        ret_partial[i] = np.sum(last_n_rewards_discounted)

        for i in reversed(range(0, ep_length - 1)):

            ret[i] = self.current_episode[i][IDX_EXTRINSIC_REWARD] + self.discount * ret[i + 1]

            last_n_rewards_discounted = last_n_rewards_discounted * self.discount
            last_n_rewards_discounted[last_n_rewards_idx] = self.current_episode[i][IDX_EXTRINSIC_REWARD]
            last_n_rewards_idx = (last_n_rewards_idx + 1) % self.n_step_n

            ret_partial[i] = np.sum(last_n_rewards_discounted)

        # Add episode to the cache
        for i in range(0, ep_length):
            self.transitions.add(self.current_episode[i][IDX_STATE], self.current_episode[i][IDX_ACTION], self.current_episode[i][IDX_EXTRINSIC_REWARD], ret[i], ret_partial[i], self.current_episode[i][IDX_TERMINAL], ep_length - 1 - i)
            i = i + 1

        self.current_episode = []


    def handle_terminal(self):

        self.moving_average_score = self.moving_average_score_mom * self.moving_average_score + (1.0 - self.moving_average_score_mom) * self.episode_score
        self.moving_average_score_clipped = self.moving_average_score_mom * self.moving_average_score_clipped + (1.0 - self.moving_average_score_mom) * self.episode_score_clipped

        self.moving_average_score_updates = self.moving_average_score_updates + 1

        zero_debiased_score = self.moving_average_score / (1.0 - self.moving_average_score_mom ** self.moving_average_score_updates)
        zero_debiased_score_clipped = self.moving_average_score_clipped / (1.0 - self.moving_average_score_mom ** self.moving_average_score_updates)

        self.score_plot.add_data_point("movingAverageScore", self.numSteps, [zero_debiased_score], False, self.show_graphs)
        self.clipped_score_plot.add_data_point("movingAverageClippedScore", self.numSteps, [zero_debiased_score_clipped], False, self.show_graphs)

        self.episode_score = 0
        self.episode_score_clipped = 0


    def perceive(self, reward:float, state:CooperativeCraftWorldState, terminal:bool, is_eval:bool):

        state = torch.from_numpy(state.getRepresentation()).float()

        if not is_eval:
            # Update the unclipped, undiscounted total reward (i.e. the game score)
            self.episode_score += reward

            # Clip the reward
            reward = np.minimum(reward, self.max_reward)
            reward = np.maximum(reward, self.min_reward)

            self.episode_score_clipped += reward

            # Store transition s, a, r, s'
            if self.lastState is not None:
            
                if self.post_episode_return_calcs_needed:
                    self.current_episode.append((self.lastState, self.lastAction, reward, self.lastTerminal))
                else:
                    self.transitions.add(self.lastState, self.lastAction, reward, 0.0, self.lastTerminal)

            if terminal:
                self.handle_terminal()

            # Necessary to process episode once lastTerminal == True so that each experience in the cache has a full return.
            if self.lastTerminal and self.post_episode_return_calcs_needed:
                self.add_episode_to_cache()

        # Select action
        actionIndex = 0
        if not terminal:
            if self.exploration_style == "e_greedy":
                actionIndex = self.eGreedy(state.to(self.device).unsqueeze(0), is_eval)
            elif self.exploration_style == "e_softmax":
                actionIndex = self.eSoftmax(state.to(self.device).unsqueeze(0), is_eval)
            else:
                print("ERROR: Unrecognised exploration_style (" + self.exploration_style + ")")
                sys.exit(0)
            
        if not is_eval:
            self.q_values_plot.add_data_point("bestq", self.numSteps, self.bestq, True, self.show_graphs)

            if self.numSteps % self.graph_save_freq == 0:
                self.q_values_plot.save_image(self.log_dir)
                self.score_plot.save_image(self.log_dir)
                self.clipped_score_plot.save_image(self.log_dir)

            self.numSteps += 1

            # Do some Q-learning updates
            if self.learn_start != -1 and self.numSteps > self.learn_start and self.numSteps % self.update_freq == 0:
                for i in range(0, self.n_replay):
                    self.learn()

            self.lastState = state.clone()
            self.lastAction = actionIndex
            self.lastTerminal = terminal

            if self.numSteps % self.target_refresh_steps == 0:
                self.refresh_target()

        return actionIndex



    def learn(self):

        assert self.transitions.size() > self.minibatch_size, 'Not enough transitions stored to learn'

        s, a, _, ret, ret_partial, s2, _, term_under_n = self.transitions.sample(self.minibatch_size)

        ret = torch.from_numpy(ret).float().to(self.device)
        ret_partial = torch.from_numpy(ret_partial).float().to(self.device)
        term_under_n = torch.from_numpy(term_under_n).float().to(self.device)
        a_tens = torch.from_numpy(a).to(self.device).unsqueeze(1).long()

        q_tp1 = self.target_network.forward(s2).detach()

        # Calculate q-values at time t
        q_values = self.network.forward(s).gather(1, a_tens).squeeze()

        value_tp1, _ = q_tp1.max(1)

        # An alternative is to calculate the greedy action first, then gather the Q-values.
        # This makes it easier to implemented Double DQN.
        # _, greedy_act = q_tp1.max(1)
        # greedy_act = greedy_act.unsqueeze(1)
        # value_tp1 = q_tp1.gather(1, greedy_act).squeeze()
        
        target_overall = torch.ones_like(term_under_n).sub(term_under_n).mul(self.discount ** self.n_step_n).mul(value_tp1).add(ret_partial)

        mixed_monte_carlo_proportion = self.mixed_monte_carlo_proportion_start * max(0, 1 - (self.numSteps / self.mixed_monte_carlo_proportion_endt))
        target_overall = target_overall.mul(1 - mixed_monte_carlo_proportion).add(ret.mul(mixed_monte_carlo_proportion))

        error = q_values - target_overall

        # Huber loss
        error.clamp_(-1.0, 1.0)

        error.div_(self.minibatch_size)

        self.optimizer.zero_grad()
        q_values.backward(error.data)
        self.optimizer.step()


    def refresh_target(self):

        self.target_network.load_state_dict(self.network.state_dict())


    def softmax(self, state):

        q = self.network.forward(state).cpu().detach().squeeze()
        self.bestq[0] = q.max().item()
        probs = F.softmax(q.div(self.softmax_temperature), dim=0)
        prob_dist = Categorical(probs)
        return prob_dist.sample().item()


    def greedy(self, state):

        q = self.network.forward(state).cpu().detach().squeeze()
        q = q.numpy()

        maxq = q[0]
        besta = [0]

        # Evaluate all other actions (with random tie-breaking)
        for a in range(1, self.n_actions):

            if q[a] > maxq:
                besta = [a]
                maxq = q[a]

            elif q[a] == maxq:
                besta.append(a)

        r = randrange(len(besta))
        action_selected = besta[r]

        self.bestq[0] = q[action_selected]

        return action_selected


    def save_model(self):

        path = self.log_dir + agent.goal_set_to_str(self.goal_set) + '.chk'
        print('Saving model to ' + path + '...')

        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
            }, path)


    def load_model(self, path):

        print('Loading model from ' + path + '...')

        checkpoint = torch.load(path, map_location=self.device)
        
        self.network = DQN(self.agent_params["dqn_config"])
        self.target_network = DQN(self.agent_params["dqn_config"])

        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.target_network.load_state_dict(self.network.state_dict())

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.agent_params["adam_lr"], betas=(self.agent_params["adam_beta1"], self.agent_params["adam_beta2"]), eps=self.agent_params["adam_eps"])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    def eSoftmax(self, state, is_eval):

        if is_eval or self.eval_mode:
            self.ep = self.eval_ep
        else:
            self.ep = self.ep_end + np.maximum(0, (self.ep_start - self.ep_end) * (self.ep_endt - np.maximum(0, self.numSteps - self.learn_start)) / self.ep_endt)
        
        a = self.softmax(state)

        # Epsilon softmax
        if np.random.uniform() < self.ep:
            return randrange(self.n_actions)
        else:
            return a
    	
    
    def eGreedy(self, state, is_eval):

        if is_eval or self.eval_mode:
            self.ep = self.eval_ep
        else:
            self.ep = self.ep_end + np.maximum(0, (self.ep_start - self.ep_end) * (self.ep_endt - np.maximum(0, self.numSteps - self.learn_start)) / self.ep_endt)
        
        a = self.greedy(state)

        # Epsilon greedy
        if np.random.uniform() < self.ep:
            return randrange(self.n_actions)
        else:
            return a


    def reset(self, agent_num, seed, possible_goal_sets, externally_visible_goal_sets):

        super().reset(agent_num, seed, possible_goal_sets, externally_visible_goal_sets)

        if self.eval_mode:

            model_file = self.saved_model_dir + self.goal + '.chk'

            checkpoint = torch.load(model_file, map_location=self.device)

            self.network = DQN(self.agent_params["dqn_config"])
            self.network.load_state_dict(checkpoint['model_state_dict'])
