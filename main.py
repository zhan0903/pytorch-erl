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
import ray
import threading,queue
from ray.rllib.utils.timer import TimerStat


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


def add_experience(state, action, next_state, reward, done, replay_buffer, replay_queue, args):
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


def evaluate(net, env, args, replay_queue, store_transition=True):
    total_reward = 0.0
    state = env.reset()
    # print(len(self.replay_buffer))
    state = utils.to_tensor(state).unsqueeze(0)
    replay_buffer = replay_memory.ReplayMemory(args.buffer_size)

    if args.is_cuda: state = state.cuda()
    done = False
    while not done:
        # if store_transition: num_frames += 1; self.gen_frames += 1
        # if render and is_render: env.render()
        action = net.forward(state)
        action.clamp(-1, 1)
        action = utils.to_numpy(action.cpu())
        # if is_action_noise: action += self.ounoise.noise()

        next_state, reward, done, info = env.step(action.flatten())  # Simulate one step in environment
        next_state = utils.to_tensor(next_state).unsqueeze(0)
        if args.is_cuda:
            next_state = next_state.cuda()
        total_reward += reward

        if store_transition:
            add_experience(state, action, next_state, reward, done, replay_buffer, replay_queue, args)

        if len(replay_buffer) > args.batch_size:
            transitions = replay_buffer.sample(args.batch_size)
            batch = replay_memory.Transition(*zip(*transitions))
            replay_queue.put(batch)

        print(len(replay_buffer))
        time.sleep(1)
        state = next_state
    # if store_transition: self.num_games += 1
    # replay_memory.put(total_reward)
    # replay_memory[key] = self.replay_buffer


class Agent:
    def __init__(self, args, env):
        self.args = args; self.env = env
        self.evolver = utils_ne.SSNE(self.args)
        self.replay_buffer = replay_memory.ReplayMemory(args.buffer_size)
        self.pop = []
        for _ in range(args.pop_size):
            self.pop.append(ddpg.Actor(args))
        for actor in self.pop: actor.eval()

        # self.workers = [Worker.remote(args) for _ in range(self.args.pop_size+1)]

        # args.is_cuda = True; args.is_memory_cuda = True
        self.rl_agent = ddpg.DDPG(args)
        self.ounoise = ddpg.OUNoise(args.action_dim)
        # self.learner = LearnerThread()
        # self.learner.start()
        # self.learning_started = False

        # Stats
        self.timers = {
            k: TimerStat()
            for k in [
            "put_weights", "get_samples", "sample_processing",
            "replay_processing", "update_priorities", "train", "sample"
        ]
        }

        self.num_games = 0; self.num_frames = 0; self.gen_frames = 0; self.len_replay = 0

    def list_argsort(self, seq):
        return sorted(range(len(seq)), key=seq.__getitem__)

    def rl_to_evo(self, rl_net, evo_net):
        for target_param, param in zip(evo_net.parameters(), rl_net.parameters()):
            target_param.data.copy_(param.data)

    def add_experience(self, state, action, next_state, reward, done, experiences):
        reward = utils.to_tensor(np.array([reward])).unsqueeze(0)
        if self.args.is_cuda: reward = reward.cuda()
        if self.args.use_done_mask:
            done = utils.to_tensor(np.array([done]).astype('uint8')).unsqueeze(0)
            if self.args.is_cuda: done = done.cuda()
        action = utils.to_tensor(action)
        if self.args.is_cuda: action = action.cuda()
        experiences.append((state, action, next_state, reward, done))
        self.replay_buffer.push(state, action, next_state, reward, done)

    def evaluate(self, net, key, replay_memory, experiences, is_render=False, is_action_noise=False, store_transition=True):
        total_reward = 0.0
        state = self.env.reset()
        # print(len(self.replay_buffer))
        state = utils.to_tensor(state).unsqueeze(0)
        if self.args.is_cuda: state = state.cuda()
        done = False
        while not done:
            if store_transition: self.num_frames += 1; self.gen_frames += 1
            if render and is_render: self.env.render()
            action = net.forward(state)
            action.clamp(-1, 1)
            action = utils.to_numpy(action.cpu())
            if is_action_noise: action += self.ounoise.noise()

            next_state, reward, done, info = self.env.step(action.flatten())  # Simulate one step in environment
            next_state = utils.to_tensor(next_state).unsqueeze(0)
            if self.args.is_cuda:
                next_state = next_state.cuda()
            total_reward += reward

            if store_transition:
                self.add_experience(state, action, next_state, reward, done, experiences)

            print(len(experiences))
            time.sleep(2)
            state = next_state
        if store_transition: self.num_games += 1
        # replay_memory.put(total_reward)
        replay_memory[key] = self.replay_buffer
        # print(total_reward)
        # print(len(self.replay_buffer))
        # return total_reward

    def train(self):
        print("begin training")
        replay_queue = mp.Queue()
        processes = []
        # with mp.Manager() as manager:
        d = mp.Manager().dict()
        q = mp.Manager().list()

        # print(len(d))
        # print(len(q))
        learner = LearnerThread(replay_queue)
        learner.start()

        for key, pop in enumerate(self.pop):
            pop.share_memory()
            p = mp.Process(target=evaluate, args=(pop, self.env, self.args, replay_queue))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            # results.append(replay_memory.get())

        # print(len(d))
        # print(len(q))
        # print(q[0])
        # print(len(q))
        # print(len(self.replay_buffer))
        exit(0)

        # for worker in self.workers: worker.set_gen_frames.remote(0)

        ####################### EVOLUTION #####################
        get_num_ids = [worker.get_gen_num.remote() for worker in self.workers]
        gen_nums = ray.get(get_num_ids)

        ##### get new experiences
        print("gen_nums:{0}".format(gen_nums))
        evaluate_ids = [worker.evaluate.remote(self.pop[key].state_dict(), self.args.num_evals)
                        for key, worker in enumerate(self.workers[:-1])]
        results_ea = ray.get(evaluate_ids)

        with self.timers["replay_processing"]:
            if self.learner.inqueue.full():
                self.num_smaples_dropped += 1
            else:
                with self.timers["get_samples"]:
                    samples = ray.get(replay)
                self.learner.inqueue.put()

        logger.debug("results:{}".format(results_ea))

        all_fitness = []

        for i in range(self.args.pop_size):
            all_fitness.append(results_ea[i][0])

        logger.debug("fitness:{}".format(all_fitness))
        best_train_fitness = max(all_fitness)
        worst_index = all_fitness.index(min(all_fitness))

        #Validation test
        champ_index = all_fitness.index(max(all_fitness))
        logger.debug("champ_index:{}".format(champ_index))

        test_score_id = self.workers[0].evaluate.remote(self.pop[champ_index].state_dict(), 5, store_transition=False)
        test_score = ray.get(test_score_id)[0]
        logger.debug("test_score:{0},champ_index:{1}".format(test_score, champ_index))

        # NeuroEvolution's probabilistic selection and recombination step
        elite_index = self.evolver.epoch(self.pop, all_fitness)
        ###################### DDPG #########################
        result_rl_id = self.workers[-1].evaluate.remote(self.rl_agent.actor.state_dict(), is_action_noise=True) #Train
        result_rl = ray.get(result_rl_id)

        logger.debug("results_rl:{}".format(result_rl))
        results_ea.append(result_rl)

        # gen_frames = 0; num_games = 0; len_replay = 0;num_frames = 0
        sum_results = np.sum(results_ea, axis=0)
        # test = sum(results_ea)
        # fitness / num_evals, len(relay_buff), self.num_frames, self.gen_frames, self.num_game
        # logger.debug("test:{0},results_ea:{1}".format(sum_results, results_ea))

        self.len_replay = sum_results[1]
        self.num_frames = sum_results[2]
        self.gen_frames = sum_results[3]
        self.num_games = sum_results[4]

        test_timer = TimerStat()
        print("gen_frames:{}".format(self.gen_frames))

        with test_timer:
            if self.len_replay > self.args.batch_size * 5:
                for _ in range(int(self.gen_frames * self.args.frac_frames_train)):
                    sample_choose = np.random.randint(self.args.pop_size+1)
                    transitions_id = self.workers[sample_choose].sample.remote(self.args.batch_size)
                    transitions = ray.get(transitions_id)
                    # transitions = results_ea[sample_choose][0].sample(self.args.batch_size)
                    batch = replay_memory.Transition(*zip(*transitions))
                    self.rl_agent.update_parameters(batch)

                # Synch RL Agent to NE
                if self.num_games % self.args.synch_period == 0:
                    self.rl_to_evo(self.rl_agent.actor, self.pop[worst_index])
                    self.evolver.rl_policy = worst_index
                    print('Synch from RL --> Nevo')

        print("test_timer:{}".format(test_timer.mean))

        return best_train_fitness, test_score, elite_index


class LearnerThread(threading.Thread):
    """Background thread that updates the local model from replay data.
    The learner thread communicates with the main thread through Queues. This
    is needed since Ray operations can only be run on the main thread. In
    addition, moving heavyweight gradient ops session runs off the main thread
    improves overall throughput.
    """
    def __init__(self,replay_queue):
        threading.Thread.__init__(self)
        # self.learner_queue_size = WindowStat("size", 50)
        # self.local_evaluator = local_evaluator
        # self.inqueue = queue.Queue(maxsize=LEARNER_QUEUE_MAX_SIZE)
        # self.outqueue = queue.Queue()
        self.queue_timer = TimerStat()
        self.grad_timer = TimerStat()
        self.daemon = True
        self.weights_updated = False
        self.stopped = False
        self.stats = {}
        self.replay_memory = replay_memory
        self.replay_queue = replay_queue

    def run(self):
        while not self.stopped:
            self.step()

    def step(self):
        if not self.replay_queue.empty():
            print("begin background training")
            print("self.replay_queue.qsize", self.replay_queue.qsize())
            print(self.replay_queue.get())
            time.sleep(1)
        else:
            print("none")
            time.sleep(1)
            return

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

        # exit(0)











