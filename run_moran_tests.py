# Imports
from axelrod.info_fsm.info_fsm import InfoFSMPlayer
from axelrod.info_fsm.info_fsm import FSMPlayer_generator
from axelrod.info_fsm.info_fsm import find_unique_FSMs
from axelrod.moran import MoranProcess
from tqdm import tqdm, trange
import pickle
import numpy as np


# Constructing the player population
one_state_players = []
for p in FSMPlayer_generator(size = 1, max_wait = 0):
    one_state_players.append(p)
print("Number of one state players: ", len(one_state_players))

two_state_players = []
for p in FSMPlayer_generator(size = 2, max_wait = 2):
    two_state_players.append(p)
print("Number of two state players: ", len(two_state_players))

unique_players = find_unique_FSMs(two_state_players) + one_state_players
print("Number of unique players: ", len(two_state_players))

players = unique_players

# Test parameters
test_repeats = 6
test_intervals = 0.2
test_max_information_cost = 1
iterations = 300
w=5
ft = lambda x: max(0, 1-w+w*x/len(players))

for info_cost in np.arange(0, test_intervals, test_max_information_cost + test_intervals):
    cacheSave = None
    for test in range(test_repeats):
        print(f"Running test {info_cost} - {test+1} ")
        mp = MoranProcess(
            players=players,
            turns=200,
            modifier=info_cost,
            fitness_transformation=ft,
            deterministic_cache = cacheSave
            births_per_iter=len(players)//20) # 5% of players die every round

        for i in trange(iterations):
            mp.next_step()

        # Speed up processing by saving the cache
        cacheSave = mp.deterministic_cache

        with open(f'pickles/mp_{int(info_cost*10)}-{test}.pickle', 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(mp, f)




