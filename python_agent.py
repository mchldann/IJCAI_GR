import os
import random
import sys
import numpy as np
import decimal
import scenario
import copy
import constants
from cooperative_craft_world import CooperativeCraftWorld
from random import randrange
from goal_recogniser import GoalRecogniser
from neural_q_learner import NeuralQLearner
from dqn import DQN_Config
from policy import eGreedyPolicy
from scheduler_agent import SchedulerAgent


ctx = decimal.Context()
ctx.prec = 20

def float_to_str(f):
    """
    Convert the given float to a string,
    without resorting to scientific notation
    """
    d1 = ctx.create_decimal(repr(f))
    return format(d1, 'f')


if len(sys.argv) < 2:
    print('Usage:', sys.argv[0], 'scenario')
    sys.exit()

if sys.argv[1] not in scenario.scenarios:
    print("Unknown scenario: " + sys.argv[1])
    sys.exit()


#####################
# ENVIRONMENT SETUP #
#####################

env_name = "cooperative_craft_world"
num_seeds = -1
max_steps = 100
size=(7, 7)
max_training_frames = 999999999

current_scenario = scenario.scenarios[sys.argv[1]]

goal_sets = current_scenario["goal_sets"]

if "externally_visible_goal_sets" not in current_scenario:
    current_scenario["externally_visible_goal_sets"] = goal_sets

if "num_spawned" not in current_scenario:
    current_scenario["num_spawned"] = scenario.scenarios["default"]["num_spawned"]

if "regeneration" not in current_scenario:
    current_scenario["regeneration"] = scenario.scenarios["default"]["regeneration"]

if "allegiance" not in current_scenario:
    current_scenario["allegiance"] = scenario.scenarios["default"]["allegiance"]

if "starting_items" not in current_scenario:
    current_scenario["starting_items"] = scenario.scenarios["default"]["starting_items"]


###################
#### SCENARIOS ####
###################

agent_params = {}

if sys.argv[1] == "train":
    agent_params["eval_mode"] = False
    n_agents = 1
    gpu = 0
else:
    agent_params["eval_mode"] = True
    n_agents = 2
    gpu = -1 # Use CPU when not training

main_rollout_style = constants.SUBGOAL_LEVEL_NON_COMMITTAL
main_kl_tol = 2.5
main_hyp_mom = 0.95
alpha = 100
beta = 10
c = 2.5
gamma_single_agent = 0.999 # 0.99
gamma = gamma_single_agent ** (1.0 / n_agents) # To maintain equivalence when there is more than one agent (since discounts are applied per agent step).

num_targets_per_item = 1

env = CooperativeCraftWorld(current_scenario, size=size, n_agents=n_agents, allow_no_op=False, render=False, ingredient_regen=current_scenario["regeneration"], max_steps=max_steps)

agent_params["agent_type"] = "dqn"

# Optimizer settings
agent_params["adam_lr"] = 0.000125
agent_params["adam_eps"] = 0.00015
agent_params["adam_beta1"] = 0.9
agent_params["adam_beta2"] = 0.999

agent_params["log_dir"] = os.path.dirname(os.path.realpath(__file__))
agent_params["log_dir"] = agent_params["log_dir"] + "/results/" + env_name + '_' + agent_params["agent_type"] + "/"
if not os.path.exists(agent_params["log_dir"]):
    os.makedirs(agent_params["log_dir"])

goal_recogniser_log_dir = agent_params["log_dir"] # Set to None to disable logging

agent_params["saved_model_dir"] = os.path.dirname(os.path.realpath(__file__)) + '/saved_models/'

eval_freq = 250000 # As per Rainbow paper
eval_steps = 125000 # As per Rainbow paper
eval_start_time = 1000000 # Don't start evaluating until gifted items at the start of training are phased out.
agent_params["eval_ep"] = 0.01
learning_curves_csv_filename = 'eval_scores.csv'
with open(agent_params["log_dir"] + learning_curves_csv_filename,'a') as fd:
    fd.write('Frame,Score\n')

agent_params["dqn_config"] = DQN_Config(env.observation_space.shape[0], env.action_space.n, gpu=gpu, noisy_nets=False, n_latent=64)

agent_params["n_step_n"] = 1
agent_params["max_reward"] = 2.0 # 1.0 # Use float("inf") for no clipping
agent_params["min_reward"] = -2.0 # -1.0 # Use float("-inf") for no clipping
agent_params["exploration_style"] = "e_greedy" # e_greedy, e_softmax
agent_params["softmax_temperature"] = 0.05
agent_params["ep_start"] = 1
agent_params["ep_end"] = 0.01
agent_params["ep_endt"] = 1000000
agent_params["discount"] = 0.99

# To help with learning from very sparse rewards initially
agent_params["mixed_monte_carlo_proportion_start"] = 0.2
agent_params["mixed_monte_carlo_proportion_endt"] = 1000000

if agent_params["eval_mode"]:
    agent_params["learn_start"] = -1 # Indicates no training
else:
    agent_params["learn_start"] = 50000

agent_params["update_freq"] = 4
agent_params["n_replay"] = 1
agent_params["minibatch_size"] = 32
agent_params["target_refresh_steps"] = 10000
agent_params["show_graphs"] = False
agent_params["graph_save_freq"] = 25000

# For training methods that require n step returns, set the below to True.
agent_params["post_episode_return_calcs_needed"] = True

transition_params = {}
transition_params["agent_params"] = agent_params
transition_params["replay_size"] = 1000000
transition_params["bufferSize"] = 512


# Training agent
agent_q_learner = NeuralQLearner("Q_learner", agent_params, transition_params)


########## PAIRED AGENTS ##########
agent_params_greedy = copy.deepcopy(agent_params)
agent_params_greedy["exploration_style"] = "e_greedy"
agent_params_greedy["eval_ep"] = 0.01
agent_q_learner_paired = NeuralQLearner("Q_learner_greedy", agent_params_greedy, transition_params)

agent_sp_paired = SchedulerAgent("SP_MCTS",
        main_rollout_style,
        None, # No goal recogniser, since using single player MCTS
        num_targets_per_item=num_targets_per_item,
        single_player=True,
        psychic=False,
        alpha=alpha,
        beta=beta,
        c=c,
        extra_rollout_stochasticity=0.0,
        external_agent_rollout_policy=eGreedyPolicy(0.1),
        gamma=gamma,
        external_agent_config=agent_params["dqn_config"])

agent_goal_rec_paired = SchedulerAgent("Goal_Recogniser",
        main_rollout_style,
        GoalRecogniser(model_temperature=0.01,
            hypothesis_momentum=main_hyp_mom,
            kl_tolerance=main_kl_tol,
            saved_model_dir=agent_params["saved_model_dir"],
            dqn_config=agent_params["dqn_config"],
            show_graph=agent_params["show_graphs"],
            log_dir = goal_recogniser_log_dir),
        num_targets_per_item=num_targets_per_item,
        single_player=False,
        psychic=False,
        alpha=alpha,
        beta=beta,
        c=c,
        extra_rollout_stochasticity=0.0,
        external_agent_rollout_policy=eGreedyPolicy(0.1),
        gamma=gamma,
        external_agent_config=agent_params["dqn_config"])

agent_i_rm_paired = SchedulerAgent("Psychic",
        main_rollout_style,
        GoalRecogniser(model_temperature=0.01,
            hypothesis_momentum=main_hyp_mom,
            kl_tolerance=main_kl_tol,
            saved_model_dir=agent_params["saved_model_dir"],
            dqn_config=agent_params["dqn_config"],
            show_graph=agent_params["show_graphs"],
            log_dir = goal_recogniser_log_dir),
        num_targets_per_item=num_targets_per_item,
        single_player=False,
        psychic=True,
        alpha=alpha,
        beta=beta,
        c=c,
        extra_rollout_stochasticity=0.0,
        external_agent_rollout_policy=eGreedyPolicy(0.1),
        gamma=gamma,
        external_agent_config=agent_params["dqn_config"])


########## BASELINES ##########
agent_sp = SchedulerAgent("SP_MCTS",
        main_rollout_style,
        None, # No goal recogniser, since using single player MCTS
        num_targets_per_item=num_targets_per_item,
        single_player=True,
        psychic=False,
        alpha=alpha,
        beta=beta,
        c=c,
        extra_rollout_stochasticity=0.0,
        external_agent_rollout_policy=eGreedyPolicy(0.1),
        gamma=gamma,
        external_agent_config=agent_params["dqn_config"])

agent_i_rm = SchedulerAgent("Psychic",
        main_rollout_style,
        GoalRecogniser(model_temperature=0.01,
            hypothesis_momentum=main_hyp_mom,
            kl_tolerance=main_kl_tol,
            saved_model_dir=agent_params["saved_model_dir"],
            dqn_config=agent_params["dqn_config"],
            show_graph=agent_params["show_graphs"],
            log_dir = goal_recogniser_log_dir),
        num_targets_per_item=num_targets_per_item,
        single_player=False,
        psychic=True,
        alpha=alpha,
        beta=beta,
        c=c,
        extra_rollout_stochasticity=0.0,
        external_agent_rollout_policy=eGreedyPolicy(0.1),
        gamma=gamma,
        external_agent_config=agent_params["dqn_config"])


agent_params_baseline = copy.deepcopy(agent_params)
agent_params_baseline["exploration_style"] = "e_greedy"
agent_params_baseline["eval_ep"] = 0.01
agent_q_learner_baseline = NeuralQLearner("Q_learner_baseline", agent_params_baseline, transition_params)


########## OUR APPROACH ##########
agent_goal_rec = SchedulerAgent("Goal_Recogniser",
        main_rollout_style,
        GoalRecogniser(model_temperature=0.01,
            hypothesis_momentum=main_hyp_mom,
            kl_tolerance=main_kl_tol,
            saved_model_dir=agent_params["saved_model_dir"],
            dqn_config=agent_params["dqn_config"],
            show_graph=agent_params["show_graphs"],
            log_dir = goal_recogniser_log_dir),
        num_targets_per_item=num_targets_per_item,
        single_player=False,
        psychic=False,
        alpha=alpha,
        beta=beta,
        c=c,
        extra_rollout_stochasticity=0.0,
        external_agent_rollout_policy=eGreedyPolicy(0.1),
        gamma=gamma,
        external_agent_config=agent_params["dqn_config"])


is_eval = False
frame_num = 0
steps_since_eval_ran = 0
steps_since_eval_began = 0
eval_total_score = 0
eval_total_episodes = 0
best_eval_average = float("-inf")
done = False

if n_agents > 1:
    agent_combos = [    
        [agent_q_learner_paired, agent_q_learner_baseline],
        [agent_q_learner_paired, agent_sp],
        [agent_q_learner_paired, agent_i_rm],
        [agent_q_learner_paired, agent_goal_rec],

        [agent_sp_paired, agent_q_learner_baseline],
        [agent_sp_paired, agent_sp],
        [agent_sp_paired, agent_i_rm],
        [agent_sp_paired, agent_goal_rec],

        [agent_goal_rec_paired, agent_q_learner_baseline],
        [agent_goal_rec_paired, agent_sp],
        [agent_goal_rec_paired, agent_i_rm],
        [agent_goal_rec_paired, agent_goal_rec],

        [agent_i_rm_paired, agent_q_learner_baseline],
        [agent_i_rm_paired, agent_sp],
        [agent_i_rm_paired, agent_i_rm],
        [agent_i_rm_paired, agent_goal_rec],
    ]
else:
    agent_combos = [[agent_q_learner]]

reward = np.zeros((n_agents), dtype=np.float32)
total_reward = np.zeros((n_agents), dtype=np.float32)

sum_total_reward = np.zeros((len(agent_combos), n_agents), dtype=np.float32)
num_trials = 0

agent_combo_idx = len(agent_combos) # So that we reset seeds during the first call of reset_all().
seed = -1
state = None

def reset_all():

    global agents, agent_combo_idx, total_reward, num_trials, seed, state

    agent_combo_idx += 1
    if agent_combo_idx >= len(agent_combos):
        agent_combo_idx = 0
        num_trials += 1

        # Only reset the environment seed when we're up to a new agent combination (to remove bias from the evaluation).
        if num_seeds != -1:
            seed = (seed + 1) % num_seeds
        else:
            seed = random.randrange(sys.maxsize)

        if agent_params["eval_mode"]:
            print("\nSetting environment seed = " + str(seed) + "...")

    agents = agent_combos[agent_combo_idx]

    total_reward = np.zeros((n_agents), dtype=np.float32)

    for i in range(len(agents)):
        agents[i].reset(i, seed, goal_sets[i], current_scenario["externally_visible_goal_sets"][i])

    for i in range(len(agents)):
        agents[i].allegiance = current_scenario["allegiance"][i]
        if n_agents == 2 and isinstance(agents[i], SchedulerAgent) and agents[i].goal_recogniser is not None:
            agents[i].goal_recogniser.set_external_agent(agents[1 - i])

    state = env.reset(agents, seed)

    # Give starting items (if applicable).
    for i in range(len(agents)):
        for item, count in current_scenario["starting_items"][i].items():
            state.inventory[i][item] = count
        
    # Make the tasks easier at the beginning of training by gifting some starting items (gradually phased out).
    if not agent_params["eval_mode"]:
        start_with_item_pr = 0.5 * max(1.0 - frame_num / 1000000, 0)
        for k, v in state.inventory[0].items():
            if np.random.uniform() < start_with_item_pr:
                state.inventory[0][k] += 1

reset_all()

if agent_params["eval_mode"]:
    results_filename = 'results_' + sys.argv[1] + '.csv'
    with open(agent_params["log_dir"] + results_filename, 'w') as fd:
        fd.write('seed,ext_agent,eval_agent,ext_agent_score,eval_agent_score\n')

while frame_num < max_training_frames:

    agent_idx = state.player_turn

    a = agents[agent_idx].perceive(reward[agent_idx], state, done, is_eval)

    # If we're in multiagent mode, and the other agent has a goal recogniser, update its goal probabilities.
    if n_agents == 2 and isinstance(agents[1 - agent_idx], SchedulerAgent) and agents[1 - agent_idx].goal_recogniser is not None:
        agents[1 - agent_idx].goal_recogniser.perceive(state, a)

    state, reward, done, info = env.step_full_state(a)

    for i in range(n_agents):
        total_reward[i] += reward[i]

    if is_eval:
        steps_since_eval_began += 1
    else:
        steps_since_eval_ran += 1
        frame_num += 1

    if done:
        if is_eval:
            print('Evaluation time step: ' + str(steps_since_eval_began) + ', episode ended with score: ' + str(total_reward[0]))
            eval_total_score += total_reward[0]
            eval_total_episodes += 1
        else:
            score_str = ''
            for i in range(0, n_agents):

                sum_total_reward[agent_combo_idx, i] += total_reward[i]
                average_total_reward = sum_total_reward[agent_combo_idx, i] / num_trials

                if agent_params["eval_mode"]:
                    score_str = score_str + ', ' + agents[i].name + ": " + str(total_reward[i]) + " (" + "{:.2f}".format(average_total_reward) + ")"
                else:
                    score_str = score_str + ', ' + agents[i].name + ": " + str(total_reward[i])

            if agent_params["eval_mode"]:
                score_str = score_str + '. Full score: ' + str(total_reward[1] + current_scenario["allegiance"][1] * total_reward[0]) + " (" + "{:.2f}".format((sum_total_reward[agent_combo_idx, 1] + current_scenario["allegiance"][1] * sum_total_reward[agent_combo_idx, 0]) / num_trials) + ")"

            print('Time step: ' + str(frame_num) + ', ep scores:' + score_str[1:])

            if agent_params["eval_mode"]:
                with open(agent_params["log_dir"] + results_filename, 'a') as fd:
                    fd.write("'" + float_to_str(seed) + ',' + agents[0].name + ',' + agents[1].name + ',' + str(total_reward[0]) + ',' + str(total_reward[1]) + '\n')

        reset_all()

        if not agent_params["eval_mode"]:
            
            if frame_num >= eval_start_time and steps_since_eval_ran >= eval_freq:
                is_eval = True
                eval_total_score = 0
                eval_total_episodes = 0

                while steps_since_eval_ran >= eval_freq:
                    steps_since_eval_ran -= eval_freq

            elif steps_since_eval_began >= eval_steps:

                ave_eval_score = float(eval_total_score) / eval_total_episodes
                print('Evaluation ended with average score of ' + str(ave_eval_score))

                if isinstance(agents[0], NeuralQLearner):
                    with open(agent_params["log_dir"] + learning_curves_csv_filename,'a') as fd:
                        fd.write(str(agents[0].numSteps) + ',' + str(ave_eval_score) + '\n')

                    if ave_eval_score > best_eval_average:
                        best_eval_average = ave_eval_score
                        print('New best eval average of ' + str(best_eval_average))
                        agents[0].save_model()
                    else:
                        print('Did not beat best eval average of ' + str(best_eval_average))

                is_eval = False
                steps_since_eval_began = 0
