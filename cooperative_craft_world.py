import copy
import math
import gym
import numpy as np
import random
from time import sleep
from vector_2d import Vector2D


_sprites = {
    "wood": '|',
    "iron": '=',
    "grass": '#',
    "gem": '@',
    "gold": '*',
    "workbench": 'W',
    "toolshed": 'T',
    "factory": 'F'
}

_max_inventory = {
    "wood": 1,
    "iron": 1,
    "grass": 1,
    "gem": 999,
    "gold": 999
}

_num_spawned = None
_reward = []

_rewardable_items = [
    "axe",
    "bed",
    "bridge",
    "cloth",
    "gem",
    "gold",
    "grass",
    "iron",
    "plank",
    "rope",
    "stick",
    "wood"
]

_recipes = {
    "axe": ("toolshed", {"iron" : 1, "stick" : 1}),
    "bed": ("workbench", {"grass" : 1, "plank" : 1}),
    "bridge": ("factory", {"iron" : 1, "wood" : 1}),
    "cloth": ("factory", {"grass" : 1}),
    "plank": ("toolshed", {"wood" : 1}),
    "rope": ("toolshed", {"grass" : 1}),
    "stick": ("workbench", {"wood" : 1})
}

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
COLLECT = 4
CRAFT = 5
NO_OP = 6


def print_or_log(str, filename):

    if filename is not None:
        with open(filename, 'a') as fd:
            fd.write(str + "\n")
    else:
        print(str)


class Screen():

    def __init__(self, size):
        self.size = size
        self.rows = []
        for y in range(0, self.size[1]):
            self.rows.append('.' * self.size[0])


    def add_sprite(self, sprite_pos, sprite):
        row = len(self.rows) - sprite_pos.y - 1
        self.rows[row] = self.rows[row][:sprite_pos.x] + sprite + self.rows[row][sprite_pos.x + 1:]


    def render(self, filename=None):
        for y in range(0, self.size[1]):
            print_or_log(self.rows[y], filename)


class CooperativeCraftWorldState():

    def __init__(self, size, action_space, n_agents=1, ingredient_regen=True, max_steps=300):
        self.player_turn = 0
        self.action_space = action_space
        self.n_agents = n_agents
        self.ingredient_regen = ingredient_regen
        self.size = size
        self.max_steps = max_steps
        self.reset()


    def step(self, action, assumed_reward_func=_reward):

        reward = [0] * self.n_agents

        if self.terminal:
            return reward

        if action == UP:
            if (self.objects["player"][self.player_turn].y + 1) < self.size[1]:
                self.objects["player"][self.player_turn].y += 1
        elif action == DOWN:
            if (self.objects["player"][self.player_turn].y - 1) >= 0:
                self.objects["player"][self.player_turn].y -= 1
        elif action == LEFT:
            if (self.objects["player"][self.player_turn].x - 1) >= 0:
                self.objects["player"][self.player_turn].x -= 1
        elif action == RIGHT:
            if (self.objects["player"][self.player_turn].x + 1) < self.size[0]:
                self.objects["player"][self.player_turn].x += 1

        # Check if we can pick up an item
        if action == COLLECT:
            for k in _max_inventory.keys():

                can_pick_up = True

                if k == "gem" and self.inventory[self.player_turn]["axe"] == 0:
                    can_pick_up = False

                if k == "gold" and self.inventory[self.player_turn]["bridge"] == 0:
                    can_pick_up = False

                if not can_pick_up:
                    continue

                if self.inventory[self.player_turn][k] < _max_inventory[k]:
                    for pos in self.objects[k]:
                        if self.objects["player"][self.player_turn].x == pos.x and self.objects["player"][self.player_turn].y == pos.y:
                            self.inventory[self.player_turn][k] += 1
                            reward[self.player_turn] += assumed_reward_func[self.player_turn][k]
                            self.objects[k].remove(pos)
                            if self.ingredient_regen:
                                self.objects[k].append(self.get_free_square())
                            break

        # Check if we can craft
        if action == CRAFT:
            for k, v in _recipes.items():
                if self.objects["player"][self.player_turn] in self.objects[v[0]]:

                    recipe_met = True
                    for ingredient, required_count in v[1].items():
                        if self.inventory[self.player_turn][ingredient] < required_count:
                            recipe_met = False
                            break

                    if recipe_met:
                        for ingredient, required_count in v[1].items():
                            self.inventory[self.player_turn][ingredient] -= required_count
                        self.inventory[self.player_turn][k] += 1
                        reward[self.player_turn] += assumed_reward_func[self.player_turn][k]

        self.steps += 1
        if self.steps >= self.max_steps:
            self.terminal = True

        self.player_turn = (self.player_turn + 1) % self.n_agents 

        return reward


    def get_object_type_at_square(self, square):
        
        if square is None:
            return "none"

        for object_type, position_list in self.objects.items():
            for pos in position_list:
                if square.x == pos.x and square.y == pos.y:
                    return object_type

        return "none"


    def is_square_free(self, square):
        for position_list in self.objects.values():
            for pos in position_list:
                if square.x == pos.x and square.y == pos.y:
                    return False
        return True


    def get_free_square(self):
        square = Vector2D(random.randrange(0, self.size[0]), random.randrange(0, self.size[1]))
        while not self.is_square_free(square):
            square = Vector2D(random.randrange(0, self.size[0]), random.randrange(0, self.size[1]))
        return square


    def getNearestObjects(self, object_name, n=1):
        p_pos = self.objects["player"][self.player_turn]
        return sorted(self.objects[object_name], key=lambda x:p_pos.distance_to(x))[0:n]


    def getObjectCount(self, object_name):
        return len(self.objects[object_name])


    def reset(self):
        self.objects = {}
        for k, v in _num_spawned.items():
            self.objects[k] = []
            for i in range(0, v):
                self.objects[k].append(self.get_free_square())

        self.inventory = []
        for _ in range(0, self.n_agents):
            agent_inv = {}
            for item in _rewardable_items:
                agent_inv[item] = 0

            self.inventory.append(agent_inv)

        self.terminal = False
        self.steps = 0


    def getRepresentation(self):
        
        rep = []
        max_dist = np.sqrt(self.size[0] * self.size[0] + self.size[1] * self.size[1])
        p_pos = self.objects["player"][self.player_turn]
        angle_increment = 45 # Note: Should divide perfectly into 360

        sorted_keys = sorted(self.objects.keys())
        for k in sorted_keys:
            if k != "player":

                # Sort by distance to the player so that nearer objects are represented earlier.
                sorted_objects = sorted(self.objects[k], key=lambda x:p_pos.distance_to(x))

                for l_bound in range(-180, 180, angle_increment):
                    u_bound = l_bound + angle_increment
                    obj_rep = 0.0
                    for obj in sorted_objects:

                        # If we're right on top of the object, represent it as a 1 in all directions
                        if obj == p_pos:
                            obj_rep = 1.0
                        else:
                            angle = int(math.degrees(math.atan2(obj.y - p_pos.y, obj.x - p_pos.x)))
                            if (angle >= l_bound and angle <= u_bound) or (angle - 360) == l_bound or (angle + 360) == u_bound:
                                obj_rep = 1.0 - p_pos.distance_to(obj) / max_dist
                                break
                    
                    rep.append(obj_rep)

        # Note: If recipes are added where required_count > 1, this will logic will need to be modified.
        sorted_keys = sorted(self.inventory[self.player_turn].keys())
        for k in sorted_keys:
            rep.append(min(self.inventory[self.player_turn][k], 1))

        rep.append(self.steps / self.max_steps)
        return np.array(rep, dtype=np.float32)


    def render(self, use_delay=False, log_dir=None):

        if log_dir is not None:
            filename = log_dir + 'state_log.txt'
        else:
            filename = None

        print_or_log("", filename)

        sorted_keys = sorted(self.inventory[0].keys())
        for k in sorted_keys:
            inv_str = k + ":"
            for agent_num in range(0, self.n_agents):
                inv_str = inv_str + " " + str(self.inventory[agent_num][k])

            print_or_log(inv_str, filename)

        print_or_log("", filename)

        screen = Screen(self.size)

        for k, v in self.objects.items():
            if k == "player":
                for agent_num in range(0, self.n_agents):
                    screen.add_sprite(v[agent_num], str(agent_num))
            else:
                for pos in v:
                    screen.add_sprite(pos, _sprites[k])

        screen.render(filename)

        if use_delay:
            sleep(0.1)


class CooperativeCraftWorld(gym.Env):
    
    def __init__(self, scenario, size=(10, 10), n_agents=1, allow_no_op=False, render=False, ingredient_regen=True, max_steps=300):

        global _num_spawned
        _num_spawned = scenario["num_spawned"]
        _num_spawned["player"] = n_agents

        self.allow_no_op = allow_no_op
        self.render = render
        self.ingredient_regen = ingredient_regen

        if self.allow_no_op:
            self.action_space = gym.spaces.Discrete(7)
        else:
            self.action_space = gym.spaces.Discrete(6)

        self.state = CooperativeCraftWorldState(size, self.action_space, n_agents=n_agents, ingredient_regen=ingredient_regen, max_steps=max_steps)

        self.observation_space = gym.spaces.Box(low=0, high=1, shape=self.state.getRepresentation().shape, dtype=np.float32)


    # This 'step' is defined just to meet the requirements of gym.Env.
    # It returns a numpy array representation of the state based on an 'eye' encoding.
    def step(self, action):
        player_turn = self.state.player_turn
        reward = self.state.step(action)[player_turn]
        info = {}

        if self.render:
            self.state.render(True)

        return self.state.getRepresentation(), reward, self.state.terminal, info
        

    # This 'step' is used by scheduler_agent
    def step_full_state(self, action):
        reward = self.state.step(action)
        info = {}

        if self.render:
            self.state.render(True)

        return self.state, reward, self.state.terminal, info


    def reset(self, agents, seed):

        # Reset the reward function
        _reward.clear()
        for agent in agents:
            reward_dic = {}

            for item in _rewardable_items:
                if item in agent.goal_set:
                    reward_dic[item] = 1
                else:
                    reward_dic[item] = 0

            _reward.append(reward_dic)

        random.seed(seed)

        self.state.reset()
        return self.state
