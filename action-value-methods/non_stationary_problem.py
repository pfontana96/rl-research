"""
Script to verify advantages of agents using Weigthed-Average action-value estimates over
agents using Sample-Average estimates in non-stationary problems, presented in Sutton 
(Reinforcement Learning 2018) p.33 ex 2.5
"""
from common.enviroments import Enviroment, NStatTestbed
from common.agents import SAAgent, WAAgent
import matplotlib.pyplot as plt
# import numpy as np

# class Agent(object):
#     """
#     Greedy agent
#     """
#     def __init__(self, N, epsilon=0, method='sample_avg', alpha=0.1):

#         self.epsilon = epsilon # if epsilon equals 0 => Greedy agent, Agent agent otherwise
#         self.N = N
#         self.method = method
#         self.alpha = alpha
#         self.reset()

#     def step(self):
#         # Step in time (choose an action)
#         prob = np.random.uniform()
#         if prob < self.epsilon:
#             # Exploratory move
#             N = self.Q.size
#             self.last_action = np.random.choice(np.array(range(N), dtype=int))
#         else:
#             # Eploitation move
#             max_actions = np.argwhere(self.Q == np.amax(self.Q)).flatten() # greedy actions (max value)
#             if len(max_actions) == 0:
#                 print('entered')
#                 self.last_action = max_actions
#             else:
#                 self.last_action = np.random.choice(max_actions)

#         return self.last_action

#     # Return string for graph legend
#     def __str__(self):
#         if self.epsilon == 0:
#             return "Greedy - " + str(self.method)
#         else:
#             return "Epsilon = " + str(self.epsilon) + " - " + str(self.method)

#     def update(self, reward):
#         """
#         Update Value function depending on reward obtained
#         """
#         self.r_sum[self.last_action] += reward
#         self.action_count[self.last_action] += 1
        
#         # Calculate alpha depending on action-value method chosen
#         if self.method == 'sample_avg':
#             alpha = 1/self.action_count[self.last_action]

#         elif self.method == 'constant':
#             alpha = self.alpha
        
#         self.Q[self.last_action] += alpha*(reward - self.Q[self.last_action]) 

#     def reset(self):
#         # Reinitialise Agent values
#         self.Q = np.zeros(self.N)
#         self.r_sum = np.zeros(self.N) # Reward Sum for each action
#         self.action_count = np.zeros(self.N) # Nb of times each action was taken
#         self.last_action = None

# class Testbed(object):
#     """
#     k-arm testbed:
#         True values of each action, q*(a) is a fixed number on start, and then each action takes
#         independent random walks (by adding a normally distributed increment with mean 0 and 
#         standard deviation 0.01)
#     """
#     def __init__(self, k):
#         self.k = k

#     def reset(self):
#         initial_value = np.random.normal(0, 1, 1)
#         self.action_rewards = np.repeat(initial_value, self.k)
    
#     def getReward(self, action):
#         # Take a step
#         delta = np.random.normal(0, 0.01, self.k)
#         self.action_rewards += delta

#         # Choose rewards for each action
#         return np.random.normal(self.action_rewards[action], 1, 1) # Normal with mean q*(a) and unit dev

#     def getOptimAction(self):
#         return np.argmax(self.action_rewards)

def main():
    k = 10
    epsilon = 0.1
    alpha = 0.1
    agents = [SAAgent(k, epsilon), WAAgent(alpha, k, epsilon)]
    testbed = NStatTestbed(k)
    iterations = 2000
    plays = 10000

    env = Enviroment(agents, testbed, iterations, plays)
    print("Running..")
    score_avg, optimals_avg = env.run()

    #Graph 1 - average score over time in independant iterations
    plt.title("10-Armed TestBed - Average Rewards")
    plt.plot(score_avg)
    plt.ylabel('Average Reward')
    plt.xlabel('Plays')
    plt.legend(agents, loc=4)
    plt.show()

    #Graph 2 - optimal selections over all plays over time in independant iterations
    plt.title("10-Armed TestBed - % Optimal Action")
    plt.plot(optimals_avg * 100)
    plt.ylim(0, 100)
    plt.ylabel('% Optimal Action')
    plt.xlabel('Plays')
    plt.legend(agents, loc=4)
    plt.show()

if __name__ == "__main__":
    main()