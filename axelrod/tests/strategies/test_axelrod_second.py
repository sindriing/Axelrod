"""Tests for the Second Axelrod strategies."""

import random

import axelrod
from .test_player import TestPlayer

C, D = axelrod.Action.C, axelrod.Action.D


class TestChampion(TestPlayer):
    name = "Champion"
    player = axelrod.Champion
    expected_classifier = {
        'memory_depth': float('inf'),
        'stochastic': True,
        'makes_use_of': set(["length"]),
        'long_run_time': False,
        'inspects_source': False,
        'manipulates_source': False,
        'manipulates_state': False
    }

    def test_strategy(self):
        # Cooperates for num_rounds / 20 (10 by default)

        actions = [(C, C), (C, D)] * 5  # Cooperate for ten rounds
        self.versus_test(axelrod.Alternator(), expected_actions=actions,
                         match_attributes={"length": 200})

        # Mirror partner for next phase
        actions += [(D, C), (C, D)] * 7  # Mirror opponent afterwards
        self.versus_test(axelrod.Alternator(), expected_actions=actions,
                         match_attributes={"length": 200})

        # Cooperate unless the opponent defected, has defected at least 40% of
        actions_1 = actions + [(D, C), (C, D), (C, C), (C, D)]
        self.versus_test(axelrod.Alternator(), expected_actions=actions_1,
                         match_attributes={"length": 200}, seed=0)

        actions_2 = actions + [(D, C), (C, D), (D, C), (C, D)]
        self.versus_test(axelrod.Alternator(), expected_actions=actions_2,
                         match_attributes={"length": 200}, seed=1)

        actions_3 = actions + [(D, C), (C, D), (C, C), (C, D)]
        self.versus_test(axelrod.Alternator(), expected_actions=actions_3,
                         match_attributes={"length": 200}, seed=2)


class TestEatherley(TestPlayer):

    name = "Eatherley"
    player = axelrod.Eatherley
    expected_classifier = {
        'memory_depth': float('inf'),
        'stochastic': True,
        'makes_use_of': set(),
        'long_run_time': False,
        'inspects_source': False,
        'manipulates_source': False,
        'manipulates_state': False
    }

    def test_strategy(self):
        # Test cooperate after opponent cooperates
        actions = [(C, C)] * 5
        self.versus_test(axelrod.Cooperator(), expected_actions=actions)

        # If opponent only defects then probability of cooperating is 0.
        actions = [(C, D), (D, D), (D, D), (D, D), (D, D)]
        self.versus_test(axelrod.Defector(), expected_actions=actions)

        # Stochastic response to defect
        actions = [(C, C), (C, D), (D, C), (C, D), (D, C)]
        self.versus_test(axelrod.Alternator(), expected_actions=actions,
                         seed=0)
        actions = [(C, C), (C, D), (C, C), (C, D), (D, C)]
        self.versus_test(axelrod.Alternator(), expected_actions=actions,
                         seed=1)

        opponent = axelrod.MockPlayer(actions=[D, C, C, D])
        actions = [(C, D), (D, C), (C, C), (C, D), (C, D)]
        self.versus_test(opponent, expected_actions=actions, seed=8)
        opponent = axelrod.MockPlayer(actions=[D, C, C, D])
        actions = [(C, D), (D, C), (C, C), (C, D), (D, D)]
        self.versus_test(opponent, expected_actions=actions, seed=2)


class TestTester(TestPlayer):

    name = "Tester"
    player = axelrod.Tester
    expected_classifier = {
        'memory_depth': float('inf'),
        'stochastic': False,
        'makes_use_of': set(),
        'long_run_time': False,
        'inspects_source': False,
        'manipulates_source': False,
        'manipulates_state': False
    }

    def test_strategy(self):
        # Alternate after 3rd round if opponent only cooperates
        actions = [(D, C)] + [(C, C), (C, C)] + [(D, C), (C, C)] * 4
        self.versus_test(axelrod.Cooperator(), expected_actions=actions,
                         attrs={"is_TFT": False})

        # Cooperate after initial defection and become TfT
        actions = [(D, C), (C, D), (C, C)]
        self.versus_test(axelrod.Alternator(), expected_actions=actions,
                         attrs={"is_TFT": True})

        # Now play TfT
        opponent = axelrod.MockPlayer(actions=[C, D, C, D, D, C])
        actions = [(D, C), (C, D), (C, C), (C, D), (D, D), (D, C), (C, C)]
        self.versus_test(opponent, expected_actions=actions,
                         attrs={"is_TFT": True})


class TestGladstein(TestPlayer):

    name = "Gladstein"
    player = axelrod.Gladstein
    expected_classifier = {
        'memory_depth': float('inf'),
        'stochastic': False,
        'makes_use_of': set(),
        'long_run_time': False,
        'inspects_source': False,
        'manipulates_source': False,
        'manipulates_state': False
    }

    def test_strategy(self):
        # Cooperates and begins to play TFT when Alternator defects
        actions = [(D, C), (C, D), (C, C), (C, D), (D, C)]
        self.versus_test(axelrod.Alternator(), expected_actions=actions,
                         attrs={'patsy': False})

        # Cooperation ratio will always be less than 0.5
        actions = [(D, C), (C, C), (C, C), (D, C), (C, C)]
        self.versus_test(axelrod.Cooperator(), expected_actions=actions,
                         attrs={'patsy': True})

        # Apologizes immediately and plays TFT
        actions = [(D, D), (C, D), (D, D), (D, D), (D, D)]
        self.versus_test(axelrod.Defector(), expected_actions=actions,
                         attrs={'patsy': False})

        # Ratio is 1/3 when MockPlayer defected for the first time.
        opponent = axelrod.MockPlayer(actions=[C, C, C, D, D])
        actions = [(D, C), (C, C), (C, C), (D, D), (C, D)]
        self.versus_test(opponent, expected_actions=actions,
                         attrs={'patsy': False})

        opponent = axelrod.AntiTitForTat()
        actions = [(D, C), (C, C), (C, D), (C, D), (D, D)]
        self.versus_test(opponent, expected_actions=actions,
                         attrs={'patsy': False})

import axelrod
from Axelrod.axelrod.tests.strategies.test_player import TestPlayer
import unittest

C, D = axelrod.Action.C, axelrod.Action.D


class TestTranquiliser(TestPlayer):
    """
 Note that this test is referred to in the documentation as an example on
 writing tests.  If you modify the tests here please also modify the
 documentation.
 """

    name = "Tit For Tat"
    player = axelrod.Tranquiliser
    expected_classifier = {
        'memory_depth': float('inf'),
        'stochastic': True,
        'makes_use_of': {"game"},
        'long_run_time': False,
        'inspects_source': False,
        'manipulates_source': False,
        'manipulates_state': False
    }

    def test_strategy(self):
        self.assertEqual(player.ratioFD2, 0)
