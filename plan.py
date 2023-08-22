import sys
import string
from abc import abstractmethod
from typing import Iterable, List
from vector_2d import Vector2D
from cooperative_craft_world import CooperativeCraftWorldState


class Plan(object):
 
    @abstractmethod
    def getNextTargets(self, state:CooperativeCraftWorldState, num_nearest_targets:int):
        pass


class AxePlan(Plan):

    def getNextTargets(self, state:CooperativeCraftWorldState, num_nearest_targets:int):

        inv = state.inventory[state.player_turn]

        if inv["stick"] > 0 and inv["iron"] > 0:
            return state.getNearestObjects("toolshed", n=num_nearest_targets)
        elif inv["stick"] > 0 and inv["iron"] == 0:
            return state.getNearestObjects("iron", n=num_nearest_targets)
        elif inv["stick"] == 0 and inv["iron"] > 0:
            return StickPlan().getNextTargets(state, num_nearest_targets)
        else:
            return merge_targets((StickPlan().getNextTargets(state, num_nearest_targets), state.getNearestObjects("iron", n=num_nearest_targets)))


class BedPlan(Plan):

    def getNextTargets(self, state:CooperativeCraftWorldState, num_nearest_targets:int):

        inv = state.inventory[state.player_turn]

        if inv["plank"] > 0 and inv["grass"] > 0:
            return state.getNearestObjects("workbench", n=num_nearest_targets)
        elif inv["plank"] > 0 and inv["grass"] == 0:
            return state.getNearestObjects("grass", n=num_nearest_targets)
        elif inv["plank"] == 0 and inv["grass"] > 0:
            return PlankPlan().getNextTargets(state, num_nearest_targets)
        else:
            return merge_targets((state.getNearestObjects("grass", n=num_nearest_targets), PlankPlan().getNextTargets(state, num_nearest_targets)))


class BridgePlan(Plan):

    def getNextTargets(self, state:CooperativeCraftWorldState, num_nearest_targets:int):

        inv = state.inventory[state.player_turn]

        if inv["iron"] > 0 and inv["wood"] > 0:
            return state.getNearestObjects("factory", n=num_nearest_targets)
        elif inv["iron"] > 0 and inv["wood"] == 0:
            return state.getNearestObjects("wood", n=num_nearest_targets)
        elif inv["iron"] == 0 and inv["wood"] > 0:
            return state.getNearestObjects("iron", n=num_nearest_targets)
        else:
            return merge_targets((state.getNearestObjects("wood", n=num_nearest_targets), state.getNearestObjects("iron", n=num_nearest_targets)))


class ClothPlan(Plan):

    def getNextTargets(self, state:CooperativeCraftWorldState, num_nearest_targets:int):

        inv = state.inventory[state.player_turn]

        if inv["grass"] == 0:
            return state.getNearestObjects("grass", n=num_nearest_targets)
        else:
            return state.getNearestObjects("factory", n=num_nearest_targets)


class GemPlan(Plan):

    def getNextTargets(self, state:CooperativeCraftWorldState, num_nearest_targets:int):

        nearest_gems = state.getNearestObjects("gem", n=num_nearest_targets)
        if len(nearest_gems) == 0:
            return []

        if state.inventory[state.player_turn]["axe"] == 0:
            return AxePlan().getNextTargets(state, num_nearest_targets)
        else:
            return nearest_gems


class GoldPlan(Plan):

    def getNextTargets(self, state:CooperativeCraftWorldState, num_nearest_targets:int):

        nearest_gold = state.getNearestObjects("gold", n=num_nearest_targets)
        if len(nearest_gold) == 0:
            return []

        if state.inventory[state.player_turn]["bridge"] == 0:
            return BridgePlan().getNextTargets(state, num_nearest_targets)
        else:
            return nearest_gold


class PlankPlan(Plan):

    def getNextTargets(self, state:CooperativeCraftWorldState, num_nearest_targets:int):

        inv = state.inventory[state.player_turn]

        if inv["wood"] == 0:
            return state.getNearestObjects("wood", n=num_nearest_targets)
        else:
            return state.getNearestObjects("toolshed", n=num_nearest_targets)


class RopePlan(Plan):

    def getNextTargets(self, state:CooperativeCraftWorldState, num_nearest_targets:int):

        inv = state.inventory[state.player_turn]

        if inv["grass"] == 0:
            return state.getNearestObjects("grass", n=num_nearest_targets)
        else:
            return state.getNearestObjects("toolshed", n=num_nearest_targets)


class StickPlan(Plan):

    def getNextTargets(self, state:CooperativeCraftWorldState, num_nearest_targets:int):

        inv = state.inventory[state.player_turn]

        if inv["wood"] == 0:
            return state.getNearestObjects("wood", n=num_nearest_targets)
        else:
            return state.getNearestObjects("workbench", n=num_nearest_targets)


class NonePlan(Plan):

    def getNextTargets(self, state:CooperativeCraftWorldState, num_nearest_targets:int):
        return []


# If an item requires more than one ingredient to craft, but one of these ingredients is missing
# then the item is uncraftable and we do not want to return a non-empty list of targets.
def merge_targets(target_lists : Iterable[Iterable[Vector2D]]) -> Iterable[Vector2D]:

    result = []
    for targets in target_lists:
        if len(targets) == 0:
            return []
        else:
            result += targets

    return result


_axe_plan = AxePlan()
_bed_plan = BedPlan()
_bridge_plan = BridgePlan()
_cloth_plan = ClothPlan()
_gem_plan = GemPlan()
_gold_plan = GoldPlan()
_plank_plan = PlankPlan()
_rope_plan = RopePlan()
_stick_plan = StickPlan()
_none_plan = NonePlan()

def str_to_plan(plan_name : string, num_nearest_targets : int) -> Plan:

    if plan_name == "axe":
        return _axe_plan
    elif plan_name == "bed":
        return _bed_plan
    elif plan_name == "bridge":
        return _bridge_plan
    elif plan_name == "cloth":
        return _cloth_plan
    elif plan_name == "gem":
        return _gem_plan
    elif plan_name == "gold":
        return _gold_plan
    elif plan_name == "plank":
        return _plank_plan
    elif plan_name == "rope":
        return _rope_plan
    elif plan_name == "stick":
        return _stick_plan
    elif plan_name == "none":
        return _none_plan
    else:
        input("ERROR: Unexpected plan name (" + plan_name + ")! Press ENTER to terminate the program.")
        sys.exit(0)
