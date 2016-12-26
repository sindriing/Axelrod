"""
Test for the Gambler strategy.
Most tests come form the LookerUp test suite
"""

import axelrod
from .test_player import TestPlayer, TestHeadsUp

import copy

C, D = axelrod.Actions.C, axelrod.Actions.D


class TestGambler(TestPlayer):

    name = "Gambler"
    player = axelrod.Gambler

    expected_classifier = {
        'memory_depth': 1,  # Default TFT table
        'stochastic': True,
        'makes_use_of': set(),
        'long_run_time': False,
        'inspects_source': False,
        'manipulates_source': False,
        'manipulates_state': False
    }

    expected_class_classifier = copy.copy(expected_classifier)
    expected_class_classifier['memory_depth'] = float('inf')

    def test_init(self):
        # Test empty table
        player = self.player(dict())
        opponent = axelrod.Cooperator()
        self.assertEqual(player.strategy(opponent), C)
        # Test default table
        tft_table = {
            ('', 'C', 'D'): 0,
            ('', 'D', 'D'): 0,
            ('', 'C', 'C'): 1,
            ('', 'D', 'C'): 1,
        }
        player = self.player(tft_table)
        opponent = axelrod.Defector()
        player.play(opponent)
        self.assertEqual(player.history[-1], C)
        player.play(opponent)
        self.assertEqual(player.history[-1], D)
        # Test malformed tables
        table = {(C, C, C): 1, ('DD', 'DD', 'C'): 1}
        with self.assertRaises(ValueError):
            player = self.player(table)


    def test_strategy(self):
        self.responses_test([C] * 4, [C, C, C, C], [C])
        self.responses_test([C] * 5, [C, C, C, C, D], [D])

    def test_defector_table(self):
        """
        Testing a lookup table that always defects if there is enough history.
        In order for the testing framework to be able to construct new player
        objects for the test, self.player needs to be callable with no
        arguments, thus we use a lambda expression which will call the
        constructor with the lookup table we want.
        """
        defector_table = {
            ('', C, D) : 0,
            ('', D, D) : 0,
            ('', C, C) : 0,
            ('', D, C) : 0,
        }
        self.player = lambda : axelrod.Gambler(defector_table)
        self.responses_test([C, C], [C, C], [D])
        self.responses_test([C, D], [D, C], [D])
        self.responses_test([D, D], [D, D], [D])


class TestPSOGambler(TestPlayer):

    name = "PSO Gambler"
    player = axelrod.PSOGambler

    expected_classifier = {
        'memory_depth': float('inf'),
        'stochastic': True,
        'makes_use_of': set(),
        'long_run_time': False,
        'inspects_source': False,
        'manipulates_source': False,
        'manipulates_state': False
    }

    def test_init(self):
        # Check for a few known keys
        known_pairs = {('CD', 'DD', 'DD'): 0.48, ('CD', 'CC', 'DD'): 0.67}
        player = self.player()
        for k, v in known_pairs.items():
            self.assertEqual(player.lookup_table[k], v)

    def test_strategy(self):
        """Starts by cooperating."""
        self.first_play_test(C)
        self.responses_test([C] * 197, [C] * 197, [C])


# Some heads up tests for PSOGambler
class PSOGamblervsDefector(TestHeadsUp):
    def test_vs(self):
        self.versus_test(axelrod.PSOGambler(), axelrod.Defector(),
                         [C, C, D, D], [D, D, D, D])


class PSOGamblervsCooperator(TestHeadsUp):
    def test_vs(self):
        self.versus_test(axelrod.PSOGambler(), axelrod.Cooperator(),
                         [C, C, C, C], [C, C, C, C])


class PSOGamblervsTFT(TestHeadsUp):
    def test_vs(self):
        self.versus_test(axelrod.PSOGambler(), axelrod.TitForTat(),
                         [C, C, C, C], [C, C, C, C])


class PSOGamblervsAlternator(TestHeadsUp):
    def test_vs(self):
        self.versus_test(axelrod.PSOGambler(), axelrod.Alternator(),
                         [C, C, D, D, D, D], [C, D, C, D, C, D])
