# Multi-Agent Intention Recognition and Progression

Source code for the IJCAI-23 paper *Multi-Agent Intention Recognition and Progression*, by Michael Dann, Yuan Yao, Natasha Alechina, Brian Logan, Felipe Meneguzzi and John Thangarajah.

## Installing the Requirements

To install the Python requirements via Anaconda, use
```conda env create -f environment.yml```.

Feel free to email Michael at michael.dann@rmit.edu.au if you have any issues getting the code to run.

## Running

To recreate the results from the paper, run:

```python python_agent.py scenario_name```

where scenario_name is one of {neutral_1, neutral_2, neutral_3, neutral_4, allied_1, allied_2, allied_3, adversarial_1, adversarial_2, adversarial_3}.

Agent scores are automatically logged to results/cooperative_craft_world_dqn/results_*scenario_name*.csv.

To train a new RL policy, run:

```python python_agent.py train```

The goal items for the RL policy can be configured in scenario.py (line 32).
