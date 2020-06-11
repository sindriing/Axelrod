# Test file
import axelrod as axl
import numpy.random as random
import numpy as np
from axelrod.action import Action
from axelrod.player import Player
from axelrod.match import Match
from axelrod.moran import MoranProcess
from axelrod.evolvable_player import EvolvablePlayer
from typing import NamedTuple

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
    action: Action
        

# Action enum to int
ato = {C:0, D:1}
    
        
class InfoFSM():
    """
    Finite State Machine with counters to allow for delayed exit
    We don't pay for information when we wait in the current state
    or make the same transition no matter the opponents move
    """
    def __init__(self, transitions: tuple) -> None:
        super().__init__()
        self._states = [State((coop, defect), wait, action)
                for coop, defect, wait, action
                in transitions]
        self._curr = self._states[0] # Assume that the initial state is always state 0
        self.modifiers = [0] # Always move blind on the first move
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

    def not_blind(self, state: State):
        return -int(state.transitions[0] != state.transitions[1])

    def current_action(self):
        return self._curr.action

    def num_states(self):
        return len(self._states)
    
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

class InfoFSMPlayer(Player):
    """Abstract base class for INFO finite state machine players."""
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
        self.fsm = InfoFSM(transitions)
    
    def strategy(self, opponent: Player) -> Action:
        if len(self.history) == 0:
            return self.fsm.current_action()
        return self.fsm.move(opponent.history[-1])

            
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

class EvolvableInfoFSM(InfoFSMPlayer, EvolvablePlayer):
    """Abstract base class for evolvable INFO finite state machine players."""
    name = "EvolvableInfoFSM"

    classifier = {
        "memory_depth": 1,
        "stochastic": False,
        "makes_use_of": set(),
        "long_run_time": False,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }

    def __init__(
        self,
        transitions: tuple = None,
        num_states: int = None,
        mutation_probability: float = 0.1,
    ) -> None:
        """If transitions is None
        then generate random parameters using num_states."""
        if transitions is None:
            self.fsm = random_FSM_uniform(num_states)
        InfoFSMPlayer.__init__(self, transitions=transitions)
        EvolvablePlayer.__init__(self)
        self.mutation_probability = mutation_probability
        self.overwrite_init_kwargs(
            transitions=transitions,
            num_states=self.fsm.num_states())

    def mutate_row(self, row):
        randoms = random.random(4)

        # Mutate Coop transition
        coop_transition = row.transitions[0]
        if randoms[0] < self.mutation_probability:
            coop_transition = random.randint(0,self.fsm.num_states())

        # Mutate Defect transition
        defect_transition = row.transitions[1]
        if randoms[1] < self.mutation_probability:
            defect_transition = random.randint(0,self.fsm.num_states())

        # Mutate wait time (increment/decrement)
        wait = row.wait
        if randoms[2] < self.mutation_probability:
            wait = np.max(0, wait + random.choice([-1,1]))

        # Mutate Action
        action = row.action
        if randoms[3] < self.mutation_probability:
            action = action.flip()

        return tuple(coop_transition, defect_transition, wait, action)

    def mutate(self):
        transitions = []

        # Mutate rows
        for row in self.fsm._states:
            transitions.append(self.mutate_row(row))

        # Shuffle rows
        if random.random() < mutation_probability:
            random.shuffle(transitions)

        return self.create_new(transitions=transitions)
