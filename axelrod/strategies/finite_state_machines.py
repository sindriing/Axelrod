import itertools
import random
from random import randrange, choice

import numpy as np

from axelrod.action import Action, UnknownActionError
from axelrod.player import EvolvablePlayer, Player

C, D = Action.C, Action.D
actions = (C, D)


def copy_lists(rows):
    new_rows = list(map(list, rows))
    return new_rows


class SimpleFSM(object):
    """Simple implementation of a finite state machine that transitions
    between states based on the last round of play.

    https://en.wikipedia.org/wiki/Finite-state_machine
    """

    def __init__(self, transitions: tuple, initial_state: int) -> None:
        """
        transitions is a list of the form
        ((state, last_opponent_action, next_state, next_action), ...)

        TitForTat would be represented with the following table:
        ((1, C, 1, C), (1, D, 1, D))
        with initial play C and initial state 1.

        """
        self._state = initial_state
        self._state_transitions = {
            (current_state, input_action): (next_state, output_action)
            for current_state, input_action, next_state, output_action in transitions
        }  # type: dict

        self._raise_error_for_bad_input()

    def _raise_error_for_bad_input(self):
        callable_states = set(
            pair[0] for pair in self._state_transitions.values()
        )
        callable_states.add(self._state)
        for state in callable_states:
            self._raise_error_for_bad_state(state)

    def _raise_error_for_bad_state(self, state: int):
        if (state, C) not in self._state_transitions or (
            state,
            D,
        ) not in self._state_transitions:
            raise ValueError(
                "state: {} does not have values for both C and D".format(state)
            )

    @property
    def state(self) -> int:
        return self._state

    @state.setter
    def state(self, new_state: int):
        self._raise_error_for_bad_state(new_state)
        self._state = new_state

    @property
    def state_transitions(self) -> dict:
        return self._state_transitions.copy()

    def transitions(self) -> list:
        return [[x[0], x[1], y[0], y[1]] for x, y in self._state_transitions.items()]

    def move(self, opponent_action: Action) -> Action:
        """Computes the response move and changes state."""
        next_state, next_action = self._state_transitions[
            (self._state, opponent_action)
        ]
        self._state = next_state
        return next_action

    def __eq__(self, other) -> bool:
        """Equality of two FSMs"""
        if not isinstance(other, SimpleFSM):
            return False
        return (self._state, self._state_transitions) == (
            other.state,
            other.state_transitions,
        )

    def num_states(self):
        """Return the number of states of the machine."""
        return len(set(state for state, action in self._state_transitions))


class FSMPlayer(Player):
    """Abstract base class for finite state machine players."""

    name = "FSM Player"

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
        transitions: tuple = ((1, C, 1, C), (1, D, 1, D)),
        initial_state: int = 1,
        initial_action: Action = C
    ) -> None:
        super().__init__()
        self.initial_state = initial_state
        self.initial_action = initial_action
        self.fsm = SimpleFSM(transitions, initial_state)

    def strategy(self, opponent: Player) -> Action:
        if len(self.history) == 0:
            return self.initial_action
        else:
            return self.fsm.move(opponent.history[-1])


class EvolvableFSMPlayer(FSMPlayer, EvolvablePlayer):
    """Abstract base class for evolvable finite state machine players."""

    name = "Evolvable FSM Player"

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
        initial_state: int = None,
        initial_action: Action = None,
        num_states: int = None,
        mutation_probability: float = 0.1,
    ) -> None:
        """If transitions, initial_state, and initial_action are None
        then generate random parameters using num_states."""
        self.mutation_probability = mutation_probability
        if not ((transitions is not None) and (initial_state is not None) and (initial_action is not None)):
            if not num_states:
                raise Exception("Insufficient Parameters to instantiate EvolvableFSMPlayer")
            transitions, initial_state, initial_action = self.random_params(num_states)
        FSMPlayer.__init__(
            self,
            transitions=transitions,
            initial_state=initial_state,
            initial_action=initial_action)
        EvolvablePlayer.__init__(self)
        self.overwrite_init_kwargs(
            transitions=transitions,
            initial_state=initial_state,
            initial_action=initial_action,
            num_states=num_states)

    @property
    def num_states(self):
        return self.fsm.num_states()

    @staticmethod
    def random_params(num_states):
        rows = []
        for j in range(num_states):
            for action in actions:
                next_state = randrange(num_states)
                next_action = choice(actions)
                row = [j, action, next_state, next_action]
                rows.append(row)
        initial_state = randrange(num_states)
        initial_action = choice(actions)
        return tuple(rows), initial_state, initial_action

    def randomize(self, num_states=None):
        if not num_states:
            num_states = self.num_states
        transitions, initial_state, initial_action = self.random_params(num_states)
        self.initial_state = initial_state
        self.initial_action = initial_action
        self.fsm = SimpleFSM(transitions, initial_state)
        self.overwrite_init_kwargs(
            transitions=transitions,
            initial_state=initial_state,
            initial_action=initial_action,
            num_states=num_states)

    @staticmethod
    def mutate_rows(rows, mutation_probability):
        rows = list(rows)
        randoms = np.random.random(len(rows))
        # Flip each value with a probability proportional to the mutation rate
        for i, row in enumerate(rows):
            if randoms[i] < mutation_probability:
                row[3] = row[3].flip()
        # Swap Two Nodes?
        if random.random() < 0.5:
            nodes = len(rows) // 2
            n1 = randrange(nodes)
            n2 = randrange(nodes)
            for j, row in enumerate(rows):
                if row[0] == n1:
                    row[0] = n2
                elif row[0] == n2:
                    row[0] = n1
            rows.sort(key=lambda x: (x[0], 0 if x[1] == C else 1))
        return rows

    def mutate(self):
        if random.random() < self.mutation_probability / 10:
            self.initial_action = self.initial_action.flip()
        if random.random() < self.mutation_probability / (10 * self.num_states):
            self.initial_state = randrange(self.num_states)
        transitions = self.mutate_rows(self.fsm.transitions(), self.mutation_probability)
        self.fsm = SimpleFSM(transitions, self.initial_state)
        self.overwrite_init_kwargs(
            transitions=transitions,
            initial_state=self.initial_state,
            initial_action=self.initial_action)
        #
        # self.init_kwargs["transitions"] = transitions
        # self.init_kwargs["initial_state"] = self.initial_state
        # self.init_kwargs["initial_action"] = self.initial_action

    @staticmethod
    def crossover_rows(rows1, rows2):
        num_states = len(rows1) // 2
        crosspoint = 2 * randrange(num_states)
        new_rows = copy_lists(rows1[:crosspoint])
        new_rows += copy_lists(rows2[crosspoint:])
        return new_rows

    def crossover(self, other):
        # Assuming that the number of states is the same
        transitions = self.crossover_rows(self.fsm.transitions(), other.fsm.transitions())
        return EvolvableFSMPlayer(
            transitions=transitions,
            initial_state=self.initial_state,
            initial_action=self.initial_action,
            mutation_probability=self.mutation_probability
        )

    def receive_vector(self, vector):
        """
        Read a serialized vector into the set of FSM parameters (less initial
        state).  Then assign those FSM parameters to this class instance.

        The vector has three parts. The first is used to define the next state
        (for each of the player's states - for each opponents action).

        The second part is the player's next moves (for each state - for
        each opponent's actions).

        Finally, a probability to determine the player's first move.
        """
        num_states = self.fsm.num_states()
        state_scale = vector[:num_states * 2]
        next_states = [int(s * (num_states - 1)) for s in state_scale]
        actions = vector[num_states * 2: -1]

        self.initial_action = C if round(vector[-1]) == 0 else D
        self.initial_state = 1

        transitions = []
        for i, (initial_state, action) in enumerate(itertools.product(range(num_states), [C, D])):
            next_action = C if round(actions[i]) == 0 else D
            transitions.append([initial_state, action, next_states[i], next_action])
        self.fsm = SimpleFSM(transitions, self.initial_state)

    def create_vector_bounds(self):
        """Creates the bounds for the decision variables."""
        size = len(self.fsm.transitions()) * 2 + 1
        lb = [0] * size
        ub = [1] * size
        return lb, ub

    @staticmethod
    def repr_rows(rows):
        ss = []
        for row in rows:
            ss.append("_".join(list(map(str, row))))
        return ":".join(ss)

    def serialize_parameters(self):
        return "{}:{}:{}:{}".format(
            self.num_states,
            self.initial_state,
            self.initial_action,
            self.repr_rows(self.fsm.transitions())
        )

    @classmethod
    def deserialize_parameters(cls, serialized):
        rows = []
        lines = serialized.split(':')
        num_states = int(lines[0])
        initial_state = int(lines[1])
        initial_action = Action.from_char(lines[2])

        for line in lines[3:]:
            row = []
            for element in line.split('_'):
                try:
                    row.append(Action.from_char(element))
                except UnknownActionError:
                    row.append(int(element))
            rows.append(row)
        return cls(
            transitions=tuple(rows),
            initial_state=initial_state,
            initial_action=initial_action,
            num_states=num_states
        )


class Fortress3(FSMPlayer):
    """Finite state machine player specified in http://DOI.org/10.1109/CEC.2006.1688322.

    Note that the description in http://www.graham-kendall.com/papers/lhk2011.pdf
    is not correct.


    Names:

    - Fortress 3: [Ashlock2006b]_
    """

    name = "Fortress3"
    classifier = {
        "memory_depth": 2,
        "stochastic": False,
        "makes_use_of": set(),
        "long_run_time": False,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }

    def __init__(self) -> None:
        transitions = (
            (1, C, 1, D),
            (1, D, 2, D),
            (2, C, 1, D),
            (2, D, 3, C),
            (3, C, 3, C),
            (3, D, 1, D),
        )

        super().__init__(
            transitions=transitions, initial_state=1, initial_action=D
        )


class Fortress4(FSMPlayer):
    """
    Finite state machine player specified in
    http://DOI.org/10.1109/CEC.2006.1688322.

    Note that the description in
    http://www.graham-kendall.com/papers/lhk2011.pdf is not correct.

    Names:

    - Fortress 4: [Ashlock2006b]_
    """

    name = "Fortress4"
    classifier = {
        "memory_depth": 3,
        "stochastic": False,
        "makes_use_of": set(),
        "long_run_time": False,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }

    def __init__(self) -> None:
        transitions = (
            (1, C, 1, D),
            (1, D, 2, D),
            (2, C, 1, D),
            (2, D, 3, D),
            (3, C, 1, D),
            (3, D, 4, C),
            (4, C, 4, C),
            (4, D, 1, D),
        )

        super().__init__(
            transitions=transitions, initial_state=1, initial_action=D
        )


class Predator(FSMPlayer):
    """
    Finite state machine player specified in
    http://DOI.org/10.1109/CEC.2006.1688322.

    Names:

    - Predator: [Ashlock2006b]_
    """

    name = "Predator"
    classifier = {
        "memory_depth": float("inf"),
        "stochastic": False,
        "makes_use_of": set(),
        "long_run_time": False,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }

    def __init__(self) -> None:
        transitions = (
            (0, C, 0, D),
            (0, D, 1, D),
            (1, C, 2, D),
            (1, D, 3, D),
            (2, C, 4, C),
            (2, D, 3, D),
            (3, C, 5, D),
            (3, D, 4, C),
            (4, C, 2, C),
            (4, D, 6, D),
            (5, C, 7, D),
            (5, D, 3, D),
            (6, C, 7, C),
            (6, D, 7, D),
            (7, C, 8, D),
            (7, D, 7, D),
            (8, C, 8, D),
            (8, D, 6, D),
        )

        super().__init__(
            transitions=transitions, initial_state=0, initial_action=C
        )


class Pun1(FSMPlayer):
    """FSM player described in [Ashlock2006]_.

    Names:

    - Pun1: [Ashlock2006]_
    """

    name = "Pun1"
    classifier = {
        "memory_depth": float("inf"),
        "stochastic": False,
        "makes_use_of": set(),
        "long_run_time": False,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }

    def __init__(self) -> None:
        transitions = ((1, C, 2, C), (1, D, 2, C), (2, C, 1, C), (2, D, 1, D))

        super().__init__(
            transitions=transitions, initial_state=1, initial_action=D
        )


class Raider(FSMPlayer):
    """
    FSM player described in http://DOI.org/10.1109/FOCI.2014.7007818.


    Names

    - Raider: [Ashlock2014]_
    """

    name = "Raider"
    classifier = {
        "memory_depth": float("inf"),
        "stochastic": False,
        "makes_use_of": set(),
        "long_run_time": False,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }

    def __init__(self) -> None:
        transitions = (
            (0, C, 2, D),
            (0, D, 2, D),
            (1, C, 1, C),
            (1, D, 1, D),
            (2, C, 0, D),
            (2, D, 3, C),
            (3, C, 0, D),
            (3, D, 1, C),
        )

        super().__init__(
            transitions=transitions, initial_state=0, initial_action=D
        )


class Ripoff(FSMPlayer):
    """
    FSM player described in http://DOI.org/10.1109/TEVC.2008.920675.

    Names

    - Ripoff: [Ashlock2008]_
    """

    name = "Ripoff"
    classifier = {
        "memory_depth": 3,
        "stochastic": False,
        "makes_use_of": set(),
        "long_run_time": False,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }

    def __init__(self) -> None:
        transitions = (
            (1, C, 2, C),
            (1, D, 3, C),
            (2, C, 1, D),
            (2, D, 3, C),
            (3, C, 3, C),  # Note that it's TFT in state 3
            (3, D, 3, D),
        )

        super().__init__(
            transitions=transitions, initial_state=1, initial_action=D
        )


class UsuallyCooperates(FSMPlayer):
    """
    This strategy cooperates except after a C following a D.

    Names:

    - Usually Cooperates (UC): [Ashlock2009]_
    """

    name = "UsuallyCooperates"
    classifier = {
        "memory_depth": float("inf"),
        "stochastic": False,
        "makes_use_of": set(),
        "long_run_time": False,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }

    def __init__(self) -> None:
        transitions = ((1, C, 1, C), (1, D, 2, C), (2, C, 1, D), (2, D, 1, C))

        super().__init__(
            transitions=transitions, initial_state=1, initial_action=C
        )


class UsuallyDefects(FSMPlayer):
    """
    This strategy defects except after a D following a C.

    Names:

    - Usually Defects (UD): [Ashlock2009]_
    """

    name = "UsuallyDefects"
    classifier = {
        "memory_depth": float("inf"),
        "stochastic": False,
        "makes_use_of": set(),
        "long_run_time": False,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }

    def __init__(self) -> None:
        transitions = ((1, C, 2, D), (1, D, 1, D), (2, C, 1, D), (2, D, 1, C))

        super().__init__(
            transitions=transitions, initial_state=1, initial_action=D
        )


class SolutionB1(FSMPlayer):
    """
    FSM player described in http://DOI.org/10.1109/TCIAIG.2014.2326012.

    Names

    - Solution B1: [Ashlock2015]_
    """

    name = "SolutionB1"
    classifier = {
        "memory_depth": 2,
        "stochastic": False,
        "makes_use_of": set(),
        "long_run_time": False,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }

    def __init__(self) -> None:
        transitions = (
            (1, C, 2, D),
            (1, D, 1, D),
            (2, C, 2, C),
            (2, D, 3, C),
            (3, C, 3, C),
            (3, D, 3, C),
        )

        super().__init__(
            transitions=transitions, initial_state=1, initial_action=D
        )


class SolutionB5(FSMPlayer):
    """

    FSM player described in http://DOI.org/10.1109/TCIAIG.2014.2326012.

    Names

    - Solution B5: [Ashlock2015]_
    """

    name = "SolutionB5"
    classifier = {
        "memory_depth": float("inf"),
        "stochastic": False,
        "makes_use_of": set(),
        "long_run_time": False,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }

    def __init__(self) -> None:
        transitions = (
            (1, C, 2, C),
            (1, D, 6, D),
            (2, C, 2, C),
            (2, D, 3, D),
            (3, C, 6, C),
            (3, D, 1, D),
            (4, C, 3, C),
            (4, D, 6, D),
            (5, C, 5, D),
            (5, D, 4, D),
            (6, C, 3, C),
            (6, D, 5, D),
        )

        super().__init__(
            transitions=transitions, initial_state=1, initial_action=D
        )


class Thumper(FSMPlayer):
    """
    FSM player described in http://DOI.org/10.1109/TEVC.2008.920675.

    Names

    - Thumper: [Ashlock2008]_
    """

    name = "Thumper"
    classifier = {
        "memory_depth": float("inf"),
        "stochastic": False,
        "makes_use_of": set(),
        "long_run_time": False,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }

    def __init__(self) -> None:
        transitions = ((1, C, 1, C), (1, D, 2, D), (2, C, 1, D), (2, D, 1, D))

        super().__init__(
            transitions=transitions, initial_state=1, initial_action=C
        )


class EvolvedFSM4(FSMPlayer):
    """
    A 4 state FSM player trained with an evolutionary algorithm.

    Names:

        - Evolved FSM 4: Original name by Marc Harper
    """

    name = "Evolved FSM 4"
    classifier = {
        "memory_depth": float("inf"),
        "stochastic": False,
        "makes_use_of": set(),
        "long_run_time": False,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }

    def __init__(self) -> None:
        transitions = (
            (0, C, 0, C),
            (0, D, 2, D),
            (1, C, 3, D),
            (1, D, 0, C),
            (2, C, 2, D),
            (2, D, 1, C),
            (3, C, 3, D),
            (3, D, 1, D),
        )

        super().__init__(
            transitions=transitions, initial_state=0, initial_action=C
        )


class EvolvedFSM16(FSMPlayer):
    """
    A 16 state FSM player trained with an evolutionary algorithm.

    Names:

        - Evolved FSM 16: Original name by Marc Harper

    """

    name = "Evolved FSM 16"
    classifier = {
        "memory_depth": float("inf"),
        "stochastic": False,
        "makes_use_of": set(),
        "long_run_time": False,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }

    def __init__(self) -> None:
        transitions = (
            (0, C, 0, C),
            (0, D, 12, D),
            (1, C, 3, D),
            (1, D, 6, C),
            (2, C, 2, D),
            (2, D, 14, D),
            (3, C, 3, D),
            (3, D, 3, D),
            (5, C, 12, D),
            (5, D, 10, D),
            (6, C, 5, C),
            (6, D, 12, D),
            (7, C, 3, D),
            (7, D, 1, C),
            (8, C, 5, C),
            (8, D, 5, C),
            (10, C, 11, D),
            (10, D, 8, C),
            (11, C, 15, D),
            (11, D, 5, D),
            (12, C, 8, C),
            (12, D, 11, D),
            (13, C, 13, D),
            (13, D, 7, D),
            (14, C, 13, D),
            (14, D, 13, D),
            (15, C, 15, D),
            (15, D, 2, C),
        )

        super().__init__(
            transitions=transitions, initial_state=0, initial_action=C
        )


class EvolvedFSM16Noise05(FSMPlayer):
    """
    A 16 state FSM player trained with an evolutionary algorithm with
    noisy matches (noise=0.05).

    Names:

        - Evolved FSM 16 Noise 05: Original name by Marc Harper
    """

    name = "Evolved FSM 16 Noise 05"
    classifier = {
        "memory_depth": float("inf"),
        "stochastic": False,
        "makes_use_of": set(),
        "long_run_time": False,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }

    def __init__(self) -> None:
        transitions = (
            (0, C, 8, C),
            (0, D, 3, D),
            (1, C, 13, C),
            (1, D, 15, D),
            (2, C, 12, C),
            (2, D, 3, D),
            (3, C, 10, C),
            (3, D, 3, D),
            (4, C, 5, D),
            (4, D, 4, D),
            (5, C, 4, D),
            (5, D, 10, D),
            (6, C, 8, C),
            (6, D, 6, D),
            (8, C, 2, C),
            (8, D, 4, D),
            (10, C, 4, D),
            (10, D, 1, D),
            (11, C, 14, D),
            (11, D, 13, C),
            (12, C, 13, C),
            (12, D, 2, C),
            (13, C, 13, C),
            (13, D, 6, C),
            (14, C, 3, D),
            (14, D, 13, D),
            (15, C, 5, D),
            (15, D, 11, C),
        )

        super().__init__(
            transitions=transitions, initial_state=0, initial_action=C
        )


# Strategies trained with Moran process objectives


class TF1(FSMPlayer):
    """
    A FSM player trained to maximize Moran fixation probabilities.

    Names:

        - TF1: Original name by Marc Harper
    """

    name = "TF1"
    classifier = {
        "memory_depth": float("inf"),
        "stochastic": False,
        "makes_use_of": set(),
        "long_run_time": False,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }

    def __init__(self) -> None:
        transitions = (
            (0, C, 7, C),
            (0, D, 1, C),
            (1, C, 11, D),
            (1, D, 11, D),
            (2, C, 8, D),
            (2, D, 8, C),
            (3, C, 3, C),
            (3, D, 12, D),
            (4, C, 6, C),
            (4, D, 3, C),
            (5, C, 11, C),
            (5, D, 8, D),
            (6, C, 13, D),
            (6, D, 14, C),
            (7, C, 4, D),
            (7, D, 2, D),
            (8, C, 14, D),
            (8, D, 8, D),
            (9, C, 0, C),
            (9, D, 10, D),
            (10, C, 8, C),
            (10, D, 15, C),
            (11, C, 6, D),
            (11, D, 5, D),
            (12, C, 6, D),
            (12, D, 9, D),
            (13, C, 9, D),
            (13, D, 8, D),
            (14, C, 8, D),
            (14, D, 13, D),
            (15, C, 4, C),
            (15, D, 5, C),
        )

        super().__init__(
            transitions=transitions, initial_state=0, initial_action=C
        )


class TF2(FSMPlayer):
    """
    A FSM player trained to maximize Moran fixation probabilities.

    Names:

        - TF2: Original name by Marc Harper
    """

    name = "TF2"
    classifier = {
        "memory_depth": float("inf"),
        "stochastic": False,
        "makes_use_of": set(),
        "long_run_time": False,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }

    def __init__(self) -> None:
        transitions = (
            (0, C, 13, D),
            (0, D, 12, D),
            (1, C, 3, D),
            (1, D, 4, D),
            (2, C, 14, D),
            (2, D, 9, D),
            (3, C, 0, C),
            (3, D, 1, D),
            (4, C, 1, D),
            (4, D, 2, D),
            (7, C, 12, D),
            (7, D, 2, D),
            (8, C, 7, D),
            (8, D, 9, D),
            (9, C, 8, D),
            (9, D, 0, D),
            (10, C, 2, C),
            (10, D, 15, C),
            (11, C, 7, D),
            (11, D, 13, D),
            (12, C, 3, C),
            (12, D, 8, D),
            (13, C, 7, C),
            (13, D, 10, D),
            (14, C, 10, D),
            (14, D, 7, D),
            (15, C, 15, C),
            (15, D, 11, D),
        )

        super().__init__(
            transitions=transitions, initial_state=0, initial_action=C
        )


class TF3(FSMPlayer):
    """
    A FSM player trained to maximize Moran fixation probabilities.

    Names:

        - TF3: Original name by Marc Harper
    """

    name = "TF3"
    classifier = {
        "memory_depth": float("inf"),
        "stochastic": False,
        "makes_use_of": set(),
        "long_run_time": False,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }

    def __init__(self) -> None:
        transitions = (
            (0, C, 0, C),
            (0, D, 3, C),
            (1, C, 5, D),
            (1, D, 0, C),
            (2, C, 3, C),
            (2, D, 2, D),
            (3, C, 4, D),
            (3, D, 6, D),
            (4, C, 3, C),
            (4, D, 1, D),
            (5, C, 6, C),
            (5, D, 3, D),
            (6, C, 6, D),
            (6, D, 6, D),
            (7, C, 7, D),
            (7, D, 5, C),
        )

        super().__init__(
            transitions=transitions, initial_state=0, initial_action=C
        )
