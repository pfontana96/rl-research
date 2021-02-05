import numpy as np 
import multiprocessing as mp
from itertools import repeat
import ctypes

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Testbeds ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class StatTestbed(object):
    """
    Stationary testbed

    k-arm testbed:
        True values of each action, q*(a), is calculated by sampling a normal dist  with mean 
        zero and unit standard deviation. The actual rewards are then calculated by sampling 
        normal distributions with mean q*(a) and unit deviation
    """
    def __init__(self, k, **kwargs):
        self.k = k
        self.mean = kwargs.get("mean", 0)

    def reset(self):
        self.action_rewards = np.random.normal(self.mean, 1, self.k) # Normal with mean 0 and unit deviation
    
    def getReward(self, action):
        return np.random.normal(self.action_rewards[action], 1, 1) # Normal with mean q*(a) and unit dev

    def getOptimAction(self):
        return np.argmax(self.action_rewards)

class NStatTestbed(StatTestbed):
    """
    Non stationary testbed

    k-arm testbed:
        True values of each action, q*(a) is a fixed number on start, and then each action takes
        independent random walks (by adding a normally distributed increment with mean 0 and 
        standard deviation 0.01)
    """

    def reset(self):
        initial_value = np.random.normal(self.mean, 1, 1)
        self.action_rewards = np.repeat(initial_value, self.k)
    
    def getReward(self, action):
        # Take a step
        delta = np.random.normal(0, 0.01, self.k)
        self.action_rewards += delta

        # Choose rewards for each action
        return np.random.normal(self.action_rewards[action], 1, 1) # Normal with mean q*(a) and unit dev

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Enviroment ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Module's functions used for multiprocessing 

class Worker(mp.Process):
    def __init__(self, name, counter, scores_buff, optimals_buff, agent_i, agents, plays, iterations, testbed):
        super(Worker, self).__init__()
        
        self.counter = counter # Shared counter
        self.scores = scores_buff # Shared array
        self.optimals = optimals_buff # Shared array
        self.agent_index = agent_i # Shared counter

        self.name = name

        self.agents = agents
        self.iterations = iterations
        self.plays = plays
        self.testbed = testbed     

    def run(self):
        n = len(self.agents)
        scores = np.zeros((self.plays, n)) # Scores of each agent
        optimals = np.zeros((self.plays, n)) # Times the agent took optimal action

        while self.counter.value <= self.iterations:
            # Print statement after every 100 iterations
            with self.counter.get_lock():
                if ((self.counter.value+1)%100) == 0:
                    print(" {name} completed iterations: {it}".format(name=self.agents[self.agent_index.value], it=self.counter.value+1))
                self.counter.value += 1

                if (self.counter.value == self.iterations) and (self.agent_index.value < (n-1)):
                    # One agent has finished but not all of them
                    self.counter.value = 0
                    with self.agent_index.get_lock():
                        self.agent_index.value += 1
            # Reset
            self.testbed.reset()
            self.agents[self.agent_index.value].reset()

            for play in range(self.plays):
                # Agent chooses an action
                action_taken = self.agents[self.agent_index.value].step()

                # Get real reward of action
                reward = self.testbed.getReward(action_taken)

                # Update agent's policy
                self.agents[self.agent_index.value].update(reward)

                scores[play, self.agent_index.value] += reward

                # Check if algorithm took optimal action
                optimal_action = self.testbed.getOptimAction()
                if action_taken == optimal_action:
                    optimals[play, self.agent_index.value] += 1

        with self.scores.get_lock():
            self.scores[:] += scores.reshape(-1)

        with self.optimals.get_lock():
            self.optimals[:] += optimals.reshape(-1)

def init_counter(counter):
    global it
    it = counter

def worker_task(agent, iterations, plays, testbed):
        
    scores = np.zeros(plays) # Scores of each agent
    optimals = np.zeros(plays) # Times the agent took optimal action

    while it.value <= iterations:
        # Print statement after every 100 iterations
        with it.get_lock():
            if ((it.value+1)%100) == 0:
                print(" {name} completed Iterations: {it}".format(name=agent, it=it.value+1))
            it.value += 1
        # Reset
        testbed.reset()
        agent.reset()

        for play in range(plays):
            # Agent chooses an action
            action_taken = agent.step()

            # Get real reward of action
            reward = testbed.getReward(action_taken)

            # Update agent's policy
            agent.update(reward)

            scores[play] += reward

            # Check if algorithm took optimal action
            optimal_action = testbed.getOptimAction()
            if action_taken == optimal_action:
                optimals[play] += 1

    return scores, optimals


class Enviroment(object):

    def __init__(self, agents, testbed, iterations, nb_plays):
        self.agents = agents
        self.testbed = testbed
        self.iterations = iterations
        self.plays = nb_plays

    def mpRun(self):
        """
        Run function using Processes instead of a Pool
        """
        n_workers = mp.cpu_count()

        it = mp.Value('i', 0)
        agent_i = mp.Value('i', 0)

        m = self.plays
        n = len(self.agents)
        scores = mp.Array(ctypes.c_double, n*m) # Shared array of scores
        optimals = mp.Array(ctypes.c_double, n*m) # Shared array of optimal choices

        workers = [Worker(str(i), it, scores, optimals, agent_i, self.agents, self.plays, self.iterations, self.testbed) for i in range(n_workers)]

        for worker in workers:
            worker.start()

        for worker in workers:
            worker.join()

        scores_avg = np.frombuffer(scores.get_obj()).reshape(m, n)
        optimals_avg = np.frombuffer(optimals.get_obj()).reshape(m, n)

        scores_avg /= self.iterations
        optimals_avg /= self.iterations

        return scores_avg, optimals_avg

    def run(self):
        """
        Run function using a Pool of processes
        """
        n_workers = mp.cpu_count()

        it = mp.Value('i', 0)
        pool = mp.Pool(n_workers, initializer=init_counter, initargs=(it,))

        scores = np.zeros((self.plays, len(self.agents)))
        optimals = np.zeros((self.plays, len(self.agents)))

        agent_ctn = 0
        for agent in self.agents:
            args = repeat((agent, self.iterations, self.plays, self.testbed), n_workers)
            scores_t, optimals_t = zip(*pool.starmap(worker_task, args))
            scores[:, agent_ctn] = np.sum(scores_t, axis=0)
            optimals[:, agent_ctn] = np.sum(optimals_t, axis=0)

            agent_ctn += 1
            it.value = 0 # Reset counter for next agent

        pool.close()
        pool.join()

        scores_avg = scores/self.iterations
        optimals_avg = optimals/self.iterations

        return scores_avg, optimals_avg
