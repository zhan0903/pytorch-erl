import numpy as np, os, time, sys, random
from core import mod_neuro_evo as utils_ne
from core import mod_utils as utils
import gym, torch
from core import replay_memory
from core import ddpg as ddpg
import argparse
import torch.multiprocessing as mp
import time
import logging
import copy
# import ray
import threading,queue
# from ray.rllib.utils.timer import TimerStat


render = False
parser = argparse.ArgumentParser()
parser.add_argument('-env', help='Environment Choices: (HalfCheetah-v2) (Ant-v2) (Reacher-v2) (Walker2d-v2) (Swimmer-v2) (Hopper-v2)', required=True)
env_tag = vars(parser.parse_args())['env']


logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.setLevel(level=logging.DEBUG)


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

original = False


def add_experience(state, action, next_state, reward, done, replay_buffer, args):
    reward = utils.to_tensor(np.array([reward])).unsqueeze(0)
    if args.is_cuda: reward = reward.cuda()
    if args.use_done_mask:
        done = utils.to_tensor(np.array([done]).astype('uint8')).unsqueeze(0)
        if args.is_cuda: done = done.cuda()
    action = utils.to_tensor(action)
    if args.is_cuda: action = action.cuda()
    # replay_buffer.appendpend((state, action, next_state, reward, done))
    # replay_queue.put((state, action, next_state, reward, done))
    replay_buffer.push(state, action, next_state, reward, done)


def evaluate(net, args, replay_queue, dict_all_returns, key, store_transition=True):
    total_reward = 0.0
    env = utils.NormalizedActions(gym.make(env_tag))
    state = env.reset()
    num_frames = 0
    state = utils.to_tensor(state).unsqueeze(0)
    replay_buffer = replay_memory.ReplayMemory(args.buffer_size)

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
            add_experience(state, action, next_state, reward, done, replay_buffer, args)

            if len(replay_buffer) > args.batch_size:
                transitions = replay_buffer.sample(args.batch_size)
                batch = replay_memory.Transition(*zip(*transitions))
                replay_queue.put(batch)
        state = next_state
    dict_all_returns[key] = (total_reward, num_frames)
    # num_frames_list.append(num_frames)

    # fitness.append(total_reward)


class Agent:
    def __init__(self, args, env):
        self.args = args
        self.evolver = utils_ne.SSNE(self.args)
        # self.replay_buffer = replay_memory.ReplayMemory(args.buffer_size)
        self.pop = []
        for _ in range(args.pop_size):
            self.pop.append(ddpg.Actor(args))
        for actor in self.pop: actor.eval()

        # self.workers = [Worker.remote(args) for _ in range(self.args.pop_size+1)]

        # args.is_cuda = True; args.is_memory_cuda = True
        self.rl_agent = ddpg.DDPG(args)
        # self.rl_agent.share_memory()

        self.ounoise = ddpg.OUNoise(args.action_dim)
        self.replay_queue = mp.Manager().Queue()  # mp.Manager().list()
        # self.replay_queue = mp.Queue()

        self.workers = self.pop.append(self.rl_agent.actor)

        self.learner = LearnerThread(self.replay_queue, self.rl_agent)
        self.learner.start()
        # Stats
        # self.timers = {
        #     k: TimerStat()
        #     for k in [
        #     "put_weights", "get_samples", "sample_processing",
        #     "replay_processing", "update_priorities", "train", "sample"
        # ]
        # }

        self.num_games = 0; self.num_frames = 0; self.gen_frames = 0; self.len_replay = 0

    def list_argsort(self, seq):
        return sorted(range(len(seq)), key=seq.__getitem__)

    def rl_to_evo(self, rl_net, evo_net):
        for target_param, param in zip(evo_net.parameters(), rl_net.parameters()):
            target_param.data.copy_(param.data)

    def train(self):
        print("begin training")
        # replay_queue = mp.Queue()
        processes = []
        # with mp.Manager() as manager:
        dict_all_returns = mp.Manager().dict()
        num_frames = mp.Manager().list()


        # print(len(d))
        # print(len(q))
        # learner = LearnerThread(replay_queue)
        # learner.start()

        time_start = time.time()
        for key, pop in enumerate(self.pop):
            pop.share_memory()
            p = mp.Process(target=evaluate, args=(pop, self.args,
                                                  self.replay_queue, dict_all_returns, key))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # exit(0)
        print("finished EA,time:",(time.time()-time_start))
        exit(0)


        ####################### EVOLUTION #####################
        # evaluate_ids = [worker.evaluate.remote(self.pop[key].state_dict(), self.args.num_evals)
        #                 for key, worker in enumerate(self.workers[:-1])]
        # results_ea = ray.get(evaluate_ids)

        # with self.timers["replay_processing"]:
        #     if self.learner.inqueue.full():
        #         self.num_smaples_dropped += 1
        #     else:
        #         with self.timers["get_samples"]:
        #             samples = ray.get(replay)
        #         self.learner.inqueue.put()
        #
        # logger.debug("results:{}".format(results_ea))
        # all_fitness

        # print(all_fitness)
        print(processes)
        print(dict_all_returns)
        all_fitness = list(dict_all_returns.values()[0])
        num_frames = list(dict_all_returns.values()[1])
        print(all_fitness)
        print(num_frames)
        self.num_frames = sum(num_frames)
        print("self.num_frames ", self.num_frames)
        print("steps", self.learner.steps)


        # for i in range(self.args.pop_size):
        #     all_fitness.append(results_ea[i][0])

        # exit(0)

        logger.debug("fitness:{}".format(all_fitness))
        best_train_fitness = max(all_fitness)
        worst_index = all_fitness.index(min(all_fitness))

        #Validation test
        champ_index = all_fitness.index(max(all_fitness))
        logger.debug("champ_index:{}".format(champ_index))

        # Validation test
        # test_score = 0.0

        # 并行实现这个
        test_return = mp.Manager().list()

        # (net, env, args, replay_queue, dict_all_fitness, num_frames_list, key, store_transition=True)
        for _ in range(5):
            p = mp.Process(target=evaluate, args=(self.pop[champ_index], self.args,
                                                  self.replay_queue,test_return,champ_index,False))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # test_score = sum(test_fitness)/5.0
        # print("test_score,",test_score)
        print("test_return,",test_return)
        exit(0)
        test_score = sum(list(test_return.values()[0])) / 5.0

        # NeuroEvolution's probabilistic selection and recombination step
        elite_index = self.evolver.epoch(self.pop, all_fitness)
        print("steps", self.learner.steps)
        # exit(0)

        ###################### DDPG #########################
        # DDPG Experience Collection
        # self.evaluate(self.rl_agent.actor, is_render=False, is_action_noise=True)  # Train

        if self.num_games % self.args.synch_period == 0:
            self.rl_to_evo(self.rl_agent.actor, self.pop[worst_index])
            self.evolver.rl_policy = worst_index
            print('Synch from RL --> Nevo')

        # print("test_timer:{}".format(test_timer.mean))

        return best_train_fitness, test_score, elite_index

# 每轮训练都新建一个线程->not need, gen_number is not need now
class LearnerThread(threading.Thread):
    """Background thread that updates the local model from replay data.
    The learner thread communicates with the main thread through Queues. This
    is needed since Ray operations can only be run on the main thread. In
    addition, moving heavyweight gradient ops session runs off the main thread
    improves overall throughput.
    """
    def __init__(self, replay_queue, rl_agent):
        threading.Thread.__init__(self)
        # self.learner_queue_size = WindowStat("size", 50)
        # self.local_evaluator = local_evaluator
        # self.inqueue = queue.Queue(maxsize=LEARNER_QUEUE_MAX_SIZE)
        # self.outqueue = queue.Queue()
        # self.queue_timer = TimerStat()
        # self.grad_timer = TimerStat()
        self.daemon = True
        self.weights_updated = False
        self.stopped = False
        self.stats = {}
        # self.replay_memory = replay_memory
        self.replay_queue = replay_queue
        self.rl_agent = rl_agent
        self.steps = 0
        self.gen_frames = 1000

    def run(self):
        while not self.stopped:
            self.step()
        # return self.rl_agent

    def step(self):
        # if not self.replay_queue.empty()  and :
        #     print("begin background training")
        #     print("self.replay_queue.qsize", self.replay_queue.qsize())
        #     print(self.replay_queue.get())
        #     print("self.replay_queue.qsize", self.replay_queue.qsize())
        #     # time.sleep(1)
        # if self.steps <= self.gen_frames:
        # print()
        print(self.replay_queue)

        # print(self.replay_queue.qsize())
        time.sleep(1)
        if not self.replay_queue.empty():
            # print("come inside")
            print(self.replay_queue.qsize())
        #     print("replay_queue,", self.replay_queue)
            batch = self.replay_queue.get()
            print("batch,", batch)
        #     self.rl_agent.update_parameters(batch)
        #     self.steps += 1
        # else:
        #     self.stopped = True

        # else:
        #     print("none")
        #     time.sleep(1)
        #     return

        # if len(self.replay_memory) is not 0:
        #     print("begin background training")
        #     time.sleep(1)
        # else:
        #     print("none")
        #     time.sleep(1)
        #     return

        # with self.queue_timer:
        #     ra, replay = self.inqueue.get()
        # if replay is not None:
        #     batch = replay.Transition(*zip(*transitions))
        #     prio_dict = {}
        #     with self.grad_timer:
        #         grad_out = self.local_evaluator.compute_apply(replay)
        #         for pid, info in grad_out.items():
        #             prio_dict[pid] = (
        #                 replay.policy_batches[pid].data.get("batch_indexes"),
        #                 info.get("td_error"))
        #             if "stats" in info:
        #                 self.stats[pid] = info["stats"]
        #     self.outqueue.put((ra, prio_dict, replay.count))
        # self.learner_queue_size.push(self.inqueue.qsize())
        # self.weights_updated = True


if __name__ == "__main__":
    parameters = Parameters()  # Create the Parameters class
    tracker = utils.Tracker(parameters, ['erl'], '_score.csv')  # Initiate tracker
    frame_tracker = utils.Tracker(parameters, ['frame_erl'], '_score.csv')  # Initiate tracker
    time_tracker = utils.Tracker(parameters, ['time_erl'], '_score.csv')
    mp.set_start_method('spawn')

    # learner = LearnerThread(self.local_evaluator)
    # learner.start()

    #Create Env
    env = utils.NormalizedActions(gym.make(env_tag))
    parameters.action_dim = env.action_space.shape[0]
    parameters.state_dim = env.observation_space.shape[0]

    logger.debug("action_dim:{0},parameters.state_dim:{1}".format(parameters.action_dim,parameters.state_dim))

    #Seed
    env.seed(parameters.seed);
    torch.manual_seed(parameters.seed); np.random.seed(parameters.seed); random.seed(parameters.seed)

    #Create Agent
    # ray.init()
    # print(torch.cuda.device_count())

    agent = Agent(parameters, env)
    print('Running', env_tag, ' State_dim:', parameters.state_dim, ' Action_dim:', parameters.action_dim)

    next_save = 100; time_start = time.time()
    while agent.num_frames <= parameters.num_frames:
        best_train_fitness, erl_score, elite_index = agent.train()
        print('#Games:', agent.num_games, '#Frames:', agent.num_frames, ' Epoch_Max:', '%.2f'%best_train_fitness if best_train_fitness != None else None, ' Test_Score:','%.2f'%erl_score if erl_score != None else None, ' Avg:','%.2f'%tracker.all_tracker[0][1], 'ENV '+env_tag)
        print('RL Selection Rate: Elite/Selected/Discarded', '%.2f'%(agent.evolver.selection_stats['elite']/agent.evolver.selection_stats['total']),
                                                             '%.2f' % (agent.evolver.selection_stats['selected'] / agent.evolver.selection_stats['total']),
                                                              '%.2f' % (agent.evolver.selection_stats['discarded'] / agent.evolver.selection_stats['total']))

        # log experiment result
        tracker.update([erl_score], agent.num_games)
        frame_tracker.update([erl_score], agent.num_frames)
        time_tracker.update([erl_score], time.time()-time_start)

        #Save Policy
        if agent.num_games > next_save:
            next_save += 100
            if elite_index != None: torch.save(agent.pop[elite_index].state_dict(), parameters.save_foldername + 'evo_net')
            print("Progress Saved")

        exit(0)











