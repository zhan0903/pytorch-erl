import torch.multiprocessing as mp
import time
from core import ddpg as ddpg
import numpy as np, os, time, sys, random
from core import mod_neuro_evo as utils_ne
from core import mod_utils as utils
import gym, torch
import argparse




render = False
parser = argparse.ArgumentParser()
parser.add_argument('-env', help='Environment Choices: (HalfCheetah-v2) (Ant-v2) (Reacher-v2) (Walker2d-v2) (Swimmer-v2) (Hopper-v2)', required=True)
env_tag = vars(parser.parse_args())['env']



def evaluate(net, args, replay_memory, dict_all_returns, key, store_transition=True):
    replay_memory.put(1)


class Parameters:
    def __init__(self):
        #Number of Frames to Run
        if env_tag == 'Hopper-v2': self.num_frames = 4000000
        elif env_tag == 'Ant-v2': self.num_frames = 6000000
        elif env_tag == 'Walker2d-v2': self.num_frames = 8000000
        else: self.num_frames = 2000000

        #USE CUDA
        self.is_cuda = True; self.is_memory_cuda = True

        #Sunchronization Period
        if env_tag == 'Hopper-v2' or env_tag == 'Ant-v2': self.synch_period = 1
        else: self.synch_period = 10

        #DDPG params
        self.use_ln = True  # True
        self.gamma = 0.99; self.tau = 0.001
        self.seed = 7
        self.batch_size = 10 # 128
        self.buffer_size = 1000000
        self.frac_frames_train = 1.0
        self.use_done_mask = True

        ###### NeuroEvolution Params ########
        #Num of trials
        if env_tag == 'Hopper-v2' or env_tag == 'Reacher-v2': self.num_evals = 5
        elif env_tag == 'Walker2d-v2': self.num_evals = 3
        else: self.num_evals = 1

        #Elitism Rate
        if env_tag == 'Hopper-v2' or env_tag == 'Ant-v2': self.elite_fraction = 0.3
        elif env_tag == 'Reacher-v2' or env_tag == 'Walker2d-v2': self.elite_fraction = 0.2
        else: self.elite_fraction = 0.1

        self.pop_size = 10
        self.crossover_prob = 0.0
        self.mutation_prob = 0.9

        #Save Results
        self.state_dim = None; self.action_dim = None #Simply instantiate them here, will be initialized later
        self.save_foldername = 'test3-debug/%s/' % env_tag
        if not os.path.exists(self.save_foldername): os.makedirs(self.save_foldername)


args = Parameters()
env = utils.NormalizedActions(gym.make(env_tag))
args.action_dim = env.action_space.shape[0]
args.state_dim = env.observation_space.shape[0]

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
print(replay_memory)
# exit(0)