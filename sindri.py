# Test file
import axelrod as axl
import random
from axelrod.action import Action
from axelrod.player import Player
from axelrod.match import Match
from axelrod.moran import MoranProcess
from typing import NamedTuple

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

C,D = Action.C, Action.D

# Formulas for FSM
blinkingTFT_formula = ((1,2,0,C),
                       (1,2,1,C),
                       (0,2,0,D))
dblblinkingTFT_formula = ((1,2,0,C),
                         (1,2,1,C),
                         (0,2,1,D))

alternator_formula = ((1,1,0,C),
                      (0,0,0,D))

double_alternator_formula = ((1,1,1,C),
                             (0,0,1,D))

cooperator_formula = ((0,0,0,C),)
defector_formula = ((0,0,0,D),)


class State(NamedTuple):
    transitions: tuple
    wait: int
    action: axl.Action
        

# Action enum to int
ato = {C:0, D:1}
    
        
class InfoFSM(axl.Player):
    """
    Finite State Machine with counters to allow for delayed exit
    We don't pay for information when we wait in the current state
    or make the same transition no matter the opponents move
    """
    name = "Info FSM Player"

    classifier = {
        "memory_depth": 1,
        "stochastic": False,
        "makes_use_of": set(),
        "long_run_time": False,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }

    def __init__(self, transitions: tuple) -> None:
        super().__init__()
        self._states = [State((coop, defect), wait, action)
                for coop, defect, wait, action
                in transitions]
        self._curr = self._states[0] # Assume that the initial state is always state 0
        self.modifiers = []
        self._counter = 0
        self._raise_error_for_bad_input()

    def move(self, opponent_action: Action) -> Action:
        """changes state then gives response"""
        if self._counter >= self._curr.wait:
            self.modifiers.append(self.not_blind(self._curr)) # Only pay for information if the transitions are different
            next_state = self._states[self._curr.transitions[ato[opponent_action]]]
            self._curr = next_state
            self._counter = 0
        else: # Ignore opponents move and stay in current state
            self.modifiers.append(0) 
            self._counter += 1
        return self._curr.action
    
    def strategy(self, opponent: Player) -> Action:
        if len(self.history) == 0:
            self.modifiers.append(0)
            return self._curr.action
        return self.move(opponent.history[-1])

    def not_blind(self, state: State):
        return -int(state.transitions[0] != state.transitions[1])
    
    def _raise_error_for_bad_input(self):
        for state in self._states:
            self._check_valid_state(state)
        
    def _check_valid_state(self, state: State):
        num_states = len(self._states)
        if (state.transitions[0] < 0
            or state.transitions[0] > num_states
            or state.transitions[1] < 0
            or state.transitions[1] > num_states
            or state.wait < 0):
            raise ValueError(f"{state} is an invalid state!")
            
            
"""Generates a random FSM of the specified number of states"""
def random_FSM_uniform(size: int, max_wait = 3):
    formula = []
    for _ in range(size):
        coop, defect = [random.randint(0, size-1) for _ in range(2)]
        wait = random.randint(0, max_wait)
        action = random.choice([C,D])
        state = (coop, defect, wait, action)
        formula.append(state)
    print(formula)
    return InfoFSM(formula)

# Strats
axl.seed(2)  # for reproducible example

bTFT = InfoFSM(blinkingTFT_formula)
dblbTFT = InfoFSM(dblblinkingTFT_formula)
myAlternator = InfoFSM(alternator_formula)
myDoubleAlternator = InfoFSM(double_alternator_formula)
cooperator = InfoFSM(cooperator_formula)
defector = InfoFSM(defector_formula)


players = [bTFT, dblbTFT, myAlternator, myDoubleAlternator, cooperator, defector]
for x in range(3):
    players.append(random_FSM_uniform(3))

# Moran Process
mp = MoranProcess(players=players, turns=200, modifier=True)
populations = mp.play()
print(mp.winning_strategy_name)
ax = mp.populations_plot()
plt.show()

# players = (dblbTFT, myDoubleAlternator)
# match = Match(players, 15)
# print(match.play())
# print(match.final_score_per_turn(), "\n")
# print(match.modified_final_score_per_turn())
# print(dblbTFT.modifiers)
