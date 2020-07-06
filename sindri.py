# Imports
# import numpy as np
# from axelrod.info_fsm.info_fsm import InfoFSMPlayer
# from axelrod.moran import MoranProcess
# import axelrod as axl
from axelrod.info_fsm.info_fsm import InfoFSMPlayer
from axelrod.info_fsm.info_fsm import FSMPlayer_generator
from axelrod.info_fsm.info_fsm import find_unique_FSMs
from axelrod.moran import MoranProcess
from tqdm import tqdm, trange
import pickle


two_state_players = []
for p in FSMPlayer_generator(size = 2, max_wait = 3):
    two_state_players.append(p)
unique_players =  find_unique_FSMs(two_state_players)
print(len(unique_players))

players = unique_players[1::5]
print(len(players))

w=5
ft = lambda x: max(0, 1-w+w*x/len(players))
# ft = lambda x: x**2
# Moran Process
mp = MoranProcess(
    players=players,
    turns=200,
    modifier=0.0,
    fitness_transformation=ft,
    births_per_iter=len(players)//20) # 5% of players die every round


for i in trange(10):
    mp.next_step()

with open('pickletest.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(mp, f)




