# Test file
import axelrod as axl
import numpy.random as random
import numpy as np
import itertools as it
from itertools import product
from axelrod.action import Action
from axelrod.player import Player
from axelrod.match import Match
from axelrod.moran import MoranProcess
from axelrod.evolvable_player import EvolvablePlayer
from typing import NamedTuple

C,D = Action.C, Action.D

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
        self._counter = 0
        self._raise_error_for_bad_input()

    def move(self, opponent_action: Action) -> Action:
        """changes state then gives response and wether it acted blind or not"""
        if self._counter >= self._curr.wait:
            pay = (not self.blind()) # Only pay for information if the transitions are different
            next_state = self._states[self._curr.transitions[ato[opponent_action]]]
            self._curr = next_state
            self._counter = 0
        else: # Ignore opponents move and stay in current state
            self._counter += 1
            pay = False
        return (self._curr.action, pay)

    def blind(self):
        return (self._curr.transitions[0] == self._curr.transitions[1])

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

    def __init__(self, transitions: tuple, name="") -> None:
        super().__init__()
        self.fsm = InfoFSM(transitions)
        if name == "":
            self.name = str(transitions)
        else:
            self.name = name

    def strategy(self, opponent: Player) -> Action:
        if len(self.history) == 0:
            return (self.fsm.current_action(), False) # First move is always blind
        return self.fsm.move(opponent.history[-1])

    """Alternative to 'strategy()'. takes a raw action rather than a player as input"""
    def _response(self, action: Action) -> Action:
        if len(self.history) == 0:
            resp =  (self.fsm.current_action(), False) # First move is always blind
        resp = self.fsm.move(action)

        s, m = resp
        self.update_history(s,action)
        return s,m

    def play(self, opponent, noise=0):
        """Overwritten from base class to work with InfoPlayer"""
        resp1, resp2 = self.strategy(opponent), opponent.strategy(self)
        s1, m1 = resp1
        s2, m2 = resp2

        self.update_history(s1, s2)
        opponent.update_history(s2, s1)
        return (s1, s2), (m1, m2)

    def __repr__(self):
        return self.name


def random_FSM_uniform(size: int, max_wait = 3, formula=False):
    """Generates a random FSM of the specified number of states"""
    formula = []
    for _ in range(size):
        coop, defect = [random.randint(0, size) for _ in range(2)]
        wait = random.randint(0, max_wait)
        action = random.choice([C,D])
        state = (coop, defect, wait, action)
        formula.append(state)
    if formula:
        return formula
    else: 
        InfoFSM(formula)


def state_generator(size = 2, max_wait = 3):
    transitions = [i for i in range(size)]
    waits = [i for i in range(max_wait)]
    actions = [C, D]
    for t1 in transitions:
        for t2 in transitions:
            for w in waits:
                for a in actions:
                    yield (t1, t2, w, a)

def FSMPlayer_generator(size = 2, max_wait = 2):
    for formula in product(*[state_generator(size ,max_wait+1) for _ in range(size)]):
        yield InfoFSMPlayer(formula)


def find_unique_FSMs(players, repeat=8):
    possible_actions = [C, D]
    RESPONSES = list(it.product(possible_actions, repeat=repeat))
    fingerprint = {}
    for p in players:
        history = []
        for resp in RESPONSES:
            p.reset()
            for action in resp:
                temp = p._response(action)
                history.append((ato[temp[0]]))
        history = tuple(history)
        if sum(history) != 0 and sum(history) != len(history):
            if history not in fingerprint:
                fingerprint[history] = p
    return list(fingerprint.values())

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
            transitions = random_FSM_uniform(size=num_states, formula=True)
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
            wait = np.max([0, wait + random.choice([-1,1])])

        # Mutate Action
        action = row.action
        if randoms[3] < self.mutation_probability:
            action = action.flip()

        return (coop_transition, defect_transition, wait, action)

    def mutate(self):
        transitions = []

        # Mutate rows
        for row in self.fsm._states:
            transitions.append(self.mutate_row(row))

        # Shuffle rows
        if random.random() < self.mutation_probability:
            random.shuffle(transitions)

        return self.create_new(transitions=transitions)
