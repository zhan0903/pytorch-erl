import multiprocessing as mp
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



# def evaluate(net, args, replay_memory, dict_all_returns, key, store_transition=True):
#     replay_memory.put(1)
#
def add_experience(state, action, next_state, reward, done, args):
    reward = utils.to_tensor(np.array([reward])).unsqueeze(0)
    if args.is_cuda: reward = reward.cuda()
    if args.use_done_mask:
        done = utils.to_tensor(np.array([done]).astype('uint8')).unsqueeze(0)
        if args.is_cuda: done = done.cuda()
    action = utils.to_tensor(action)
    if args.is_cuda: action = action.cuda()
    # replay_buffer.append(state, action, next_state, reward, done)
    # replay_queue.put((state, action, next_state, reward, done))
    # print("before put")
    return (state, action, next_state, reward, done)


def evaluate(net, args, replay_memory, dict_all_returns, key, store_transition=True):
    total_reward = 0.0
    env = utils.NormalizedActions(gym.make(env_tag))
    state = env.reset()
    num_frames = 0
    state = utils.to_tensor(state).unsqueeze(0)
    # replay_buffer = replay_memory.ReplayMemory(args.buffer_size)
    # replay_memory[key] = replay_memory

    if args.is_cuda: state = state.cuda()
    done = False
    while not done:
        if store_transition: num_frames += 1
        # if render and is_render: env.render()
        action = net.forward(state)
        action.clamp(-1, 1)
        action = utils.to_numpy(action.cpu())
        # if is_action_noise: action += self.ounoise.noise()
        # print("1")

        next_state, reward, done, info = env.step(action.flatten())  # Simulate one step in environment
        next_state = utils.to_tensor(next_state).unsqueeze(0)
        if args.is_cuda:
            next_state = next_state.cuda()
        total_reward += reward

        if store_transition:
            print(add_experience(state, action, next_state, reward, done,args))
            replay_memory.put_nowait(add_experience(state, action, next_state, reward, done,args))
            print("dfadfasfdsf")
            print("done:",done)
            # replay_memory[key] = replay_memory

            # if len(replay_buffer) > args.batch_size:
            #     transitions = replay_buffer.sample(args.batch_size)
            #     batch = replay_memory.Transition(*zip(*transitions))
            #     replay_queue.put(batch)
        state = next_state


class Parameters:
    def __init__(self):
        #Number of Frames to Run
        if env_tag == 'Hopper-v2': self.num_frames = 4000000
        elif env_tag == 'Ant-v2': self.num_frames = 6000000
        elif env_tag == 'Walker2d-v2': self.num_frames = 8000000
        else: self.num_frames = 2000000

        #USE CUDA
        self.is_cuda = False; self.is_memory_cuda = True

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

        self.pop_size = 1
        self.crossover_prob = 0.0
        self.mutation_prob = 0.9

        #Save Results
        self.state_dim = None; self.action_dim = None #Simply instantiate them here, will be initialized later
        self.save_foldername = 'test3-debug/%s/' % env_tag
        if not os.path.exists(self.save_foldername): os.makedirs(self.save_foldername)




if __name__ == "__main__":
    args = Parameters()
    env = utils.NormalizedActions(gym.make(env_tag))
    args.action_dim = env.action_space.shape[0]
    args.state_dim = env.observation_space.shape[0]

    pop = []
    for _ in range(1):
        pop.append(ddpg.Actor(args))
    for actor in pop: actor.eval()

    mp.set_start_method('spawn')

    replay_memory = mp.Queue()
    # dict_all_returns = mp.Manager().dict()
    processes = []

    time_start = time.time()
    for key, pop in enumerate(pop):
        pop.share_memory()
        p = mp.Process(target=evaluate, args=(pop, args, replay_memory
                                              , None, key))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # exit(0)
    print("finished EA,time:", (time.time() - time_start))
    print(replay_memory)
    # exit(0)