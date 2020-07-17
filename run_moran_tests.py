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
print("Number of unique players: ", len(unique_players))

players = unique_players

# Test parameters
test_repeats = 1
test_intervals = 0.2
test_min_information_cost = 0.4
test_max_information_cost = 0.8
iterations = 50
w=5
ft = lambda x: max(0, 1-w+w*x/len(players))

for info_cost in np.arange(test_min_information_cost, test_max_information_cost + test_intervals, test_intervals):
    for test in range(test_repeats):
        print(f"Running test {info_cost} - {test+1} ")
        if test == 0:
            mp = MoranProcess(
                players=players,
                turns=200,
                modifier=info_cost,
                extra_statistics=True,
                fitness_transformation=ft,
                births_per_iter=len(players)//20) # 5% of players die every round
        else:
            # Reset is quicker than creating a new process
            mp.reset()

        for i in trange(iterations):
            if not mp.next_step():
                 print("Population has fixated")
                 break

        with open(f'pickles/mp_cost-{int(info_cost*100)}_num-{test}.pickle', 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(mp, f)




