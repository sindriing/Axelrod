# Imports
from axelrod.info_fsm.info_fsm import InfoFSMPlayer
from axelrod.info_fsm.info_fsm import FSMPlayer_generator
from axelrod.info_fsm.info_fsm import generate_unique_FSMs
from axelrod.moran import MoranProcess
from tqdm import tqdm, trange
import pickle
import random   
import numpy as np


# Constructing the player population

one_state_players = []
for p in FSMPlayer_generator(size = 1, max_wait = 0):
    one_state_players.append(p)
print("Number of one state players: ", len(one_state_players))

three_state_players = []
for p in FSMPlayer_generator(size = 3, max_wait = 2):
    three_state_players.append(p)
print("Number of three state players: ", len(three_state_players))

random.shuffle(three_state_players)
temp_players = one_state_players + three_state_players

unique_players = []
for p in generate_unique_FSMs(temp_players):
    unique_players.append(p)
    if len(unique_players) >= 64:
        break
players = unique_players
print("Number of players in population: ", len(players))

# Test parameters
test_repeats = 15
test_intervals = 0.5 
test_min_information_cost = 0.0
test_max_information_cost = 7.0
iterations = 500
w=5
ft = lambda x: max(0, 1-w+w*x)

for info_cost in np.arange(test_min_information_cost, test_max_information_cost + test_intervals, test_intervals):
    for test in range(test_repeats):
        print(f"Running test {int(info_cost)} - {test+1} ")
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
            mp.births_per_iter=len(players)//20 

        # Run the simulation
        for i in trange(iterations):
            if i>0 and i%100==0:
                mp.births_per_iter += 1
            if not mp.next_step():
                print("Population has fixated")
                break
        print("Top 3 players")
        for i, p in enumerate(mp.populations[-1].most_common()):
            print(p)
            if i >= 2:
                break

        with open(f'pickles/mp_threestate-{int(info_cost*100)}_num-{test}.pickle', 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(mp, f)




