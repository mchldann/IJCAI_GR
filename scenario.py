import constants


scenarios = {}


# Defaults
scenarios["default"] = {}

scenarios["default"]["regeneration"] = False

scenarios["default"]["allegiance"] = [constants.SELFISH, constants.SELFISH]

scenarios["default"]["starting_items"] = [{}, {}]

scenarios["default"]["num_spawned"] = {
    "wood": 4,
    "iron": 4,
    "grass": 4,
    "gem": 2,
    "gold": 2,
    "workbench": 2,
    "toolshed": 2,
    "factory": 2,
}


# RL training scenario
scenarios["train"] = {}

scenarios["train"]["goal_sets"] = [
    ["cloth", "stick", "plank"]
]


#########################
### NEUTRAL SCENARIOS ###
#########################

# Neutral scenario 1

scenarios["neutral_1"] = {}

scenarios["neutral_1"]["goal_sets"] = [
    ["gem"],
    ["gem", "gold"]
]

scenarios["neutral_1"]["externally_visible_goal_sets"] = [
    ["gem", "gold"],
    ["gem", "gold"]
]

scenarios["neutral_1"]["num_spawned"] = {
    "wood": 2,
    "iron": 2,
    "grass": 2,
    "gem": 5,
    "gold": 4,
    "workbench": 1,
    "toolshed": 1,
    "factory": 1,
}


# Neutral scenario 2

scenarios["neutral_2"] = {}

scenarios["neutral_2"]["goal_sets"] = [
    ["plank", "stick"],
    ["bridge", "gold", "rope"]
]

scenarios["neutral_2"]["externally_visible_goal_sets"] = [
    ["stick", "plank", "rope", "cloth"],
    ["bridge", "gold", "rope"]
]

scenarios["neutral_2"]["num_spawned"] = {
    "wood": 2,
    "iron": 2,
    "grass": 1,
    "gem": 0,
    "gold": 3,
    "workbench": 2,
    "toolshed": 2,
    "factory": 2,
}


# Neutral scenario 3

scenarios["neutral_3"] = {}

scenarios["neutral_3"]["goal_sets"] = [
    ["plank", "stick"],
    ["bridge", "gold", "rope"]
]

scenarios["neutral_3"]["externally_visible_goal_sets"] = [
    ["cloth", "plank"],
    ["bridge", "gold", "rope"]
]

scenarios["neutral_3"]["num_spawned"] = {
    "wood": 2,
    "iron": 2,
    "grass": 1,
    "gem": 0,
    "gold": 3,
    "workbench": 2,
    "toolshed": 2,
    "factory": 2,
}


# Neutral scenario 4

scenarios["neutral_4"] = {}

scenarios["neutral_4"]["goal_sets"] = [
    ["stick"],
    ["bridge", "gold", "rope"]
]

scenarios["neutral_4"]["externally_visible_goal_sets"] = [
    ["cloth", "plank"],
    ["bridge", "gold", "rope"]
]

scenarios["neutral_4"]["num_spawned"] = {
    "wood": 2,
    "iron": 2,
    "grass": 1,
    "gem": 0,
    "gold": 3,
    "workbench": 2,
    "toolshed": 2,
    "factory": 2,
}


########################
### ALLIED SCENARIOS ###
########################

# Allied scenario 1

scenarios["allied_1"] = {}

scenarios["allied_1"]["goal_sets"] = [
    ["bed", "gold"],
    ["axe", "bed"]
]

scenarios["allied_1"]["externally_visible_goal_sets"] = [
    ["gold", "cloth", "bed"],
    ["axe", "bed"]
]

scenarios["allied_1"]["allegiance"] = [constants.ALLIED, constants.ALLIED]

scenarios["allied_1"]["num_spawned"] = {
    "wood": 1,
    "iron": 1,
    "grass": 1,
    "gem": 3,
    "gold": 3,
    "workbench": 2,
    "toolshed": 2,
    "factory": 2,
}


# Allied scenario 2

scenarios["allied_2"] = {}

scenarios["allied_2"]["goal_sets"] = [
    ["cloth", "gold"],
    ["cloth", "gold", "stick"]
]

scenarios["allied_2"]["externally_visible_goal_sets"] = [
    ["cloth", "gold", "gem", "stick"],
    ["cloth", "gold", "stick"]
]

scenarios["allied_2"]["allegiance"] = [constants.ALLIED, constants.ALLIED]

scenarios["allied_2"]["num_spawned"] = {
    "wood": 2,
    "iron": 2,
    "grass": 1,
    "gem": 2,
    "gold": 2,
    "workbench": 1,
    "toolshed": 1,
    "factory": 1,
}


# Allied scenario 3

scenarios["allied_3"] = {}

scenarios["allied_3"]["goal_sets"] = [
    ["bridge", "cloth", "rope"],
    ["axe", "bed"]
]

scenarios["allied_3"]["externally_visible_goal_sets"] = [
    ["axe", "bridge", "cloth", "plank", "rope", "stick"],
    ["axe", "bed"]
]

scenarios["allied_3"]["allegiance"] = [constants.ALLIED, constants.ALLIED]

scenarios["allied_3"]["num_spawned"] = {
    "wood": 2,
    "iron": 2,
    "grass": 2,
    "gem": 3,
    "gold": 3,
    "workbench": 2,
    "toolshed": 2,
    "factory": 2,
}


#############################
### ADVERSARIAL SCENARIOS ###
#############################

# Adversarial scenario 1

scenarios["adversarial_1"] = {}

scenarios["adversarial_1"]["goal_sets"] = [
    ["rope"],
    ["axe", "bed", "gold"]
]

scenarios["adversarial_1"]["externally_visible_goal_sets"] = [
    ["axe", "bridge", "rope"],
    ["axe", "bed", "gold"]
]

scenarios["adversarial_1"]["allegiance"] = [constants.ADVERSARIAL, constants.ADVERSARIAL]

scenarios["adversarial_1"]["num_spawned"] = {
    "wood": 4,
    "iron": 4,
    "grass": 4,
    "gem": 1,
    "gold": 1,
    "workbench": 3,
    "toolshed": 3,
    "factory": 3,
}


# Adversarial scenario 2

scenarios["adversarial_2"] = {}

scenarios["adversarial_2"]["goal_sets"] = [
    ["gem", "gold"],
    ["axe", "bridge", "cloth", "rope"]
]

scenarios["adversarial_2"]["externally_visible_goal_sets"] = [
    ["cloth", "gem", "gold", "rope"],
    ["axe", "bridge", "cloth", "rope"]
]

scenarios["adversarial_2"]["allegiance"] = [constants.ADVERSARIAL, constants.ADVERSARIAL]

scenarios["adversarial_2"]["num_spawned"] = {
    "wood": 1,
    "iron": 4,
    "grass": 2,
    "gem": 2,
    "gold": 2,
    "workbench": 2,
    "toolshed": 2,
    "factory": 2,
}


# Adversarial scenario 3

scenarios["adversarial_3"] = {}

scenarios["adversarial_3"]["goal_sets"] = [
    ["axe", "bed", "bridge"],
    ["cloth", "plank", "rope", "stick"]
]

scenarios["adversarial_3"]["externally_visible_goal_sets"] = [
    ["axe", "bed", "bridge", "cloth", "rope"],
    ["cloth", "plank", "rope", "stick"]
]

scenarios["adversarial_3"]["allegiance"] = [constants.ADVERSARIAL, constants.ADVERSARIAL]

scenarios["adversarial_3"]["num_spawned"] = {
    "wood": 3,
    "iron": 6,
    "grass": 3,
    "gem": 2,
    "gold": 2,
    "workbench": 2,
    "toolshed": 4,
    "factory": 2,
}
