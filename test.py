import torch.multiprocessing as mp
import time
from core import ddpg as ddpg
import numpy as np, os, time, sys, random
from core import mod_neuro_evo as utils_ne
from core import mod_utils as utils
import gym, torch




def evaluate(net, args, replay_memory, dict_all_returns, key, store_transition=True):
    replay_memory.put(1)



class Parameters:
    def __init__(self):
        self.is_cuda = True
        self.is_memory_cuda = True
        self.pop_size = 10
        self.state_dim = 8
        self.action_dim = 2
        self.use_ln = True
        self.num_evals = 1



args = Parameters()
pop = []
for _ in range(10):
    pop.append(ddpg.Actor(args))
for actor in pop: actor.eval()



replay_memory = mp.Queue()
dict_all_returns = mp.Manager().dict()
processes = []

time_start = time.time()
for key, pop in enumerate(pop):
    pop.share_memory()
    p = mp.Process(target=evaluate, args=(pop, args, replay_memory
                                          , dict_all_returns, key))
    p.start()
    processes.append(p)
    # time.sleep(10)
    # print(self.replay_memory.get())

for p in processes:
    p.join()

# exit(0)
print("finished EA,time:", (time.time() - time_start))
# exit(0)