"""
Script to verify advantages of e-Greedy agents over Greedy agents, both using sample-average
action-value estimates in stationary problems, presented in Sutton (Reinforcement Learning 
2018) p.29
"""


import numpy as np
import matplotlib.pyplot as plt

class Agent(object):
    """
    Greedy agent
    """
    def __init__(self, N, epsilon=0):

        self.epsilon = epsilon # if epsilon equals 0 => Greedy agent, Agent agent otherwise
        self.N = N
        self.reset()

    def step(self):
        # Step in time (choose an action)
        prob = np.random.uniform()
        if prob < self.epsilon:
            # Exploratory move
            N = self.Q.size
            self.last_action = np.random.choice(np.array(range(N), dtype=int))
        else:
            # Eploitation move
            max_actions = np.argwhere(self.Q == np.amax(self.Q)).flatten() # greedy actions (max value)
            if len(max_actions) == 0:
                print('entered')
                self.last_action = max_actions
            else:
                self.last_action = np.random.choice(max_actions)

        return self.last_action

    # Return string for graph legend
    def __str__(self):
        if self.epsilon == 0:
            return "Greedy"
        else:
            return "Epsilon = " + str(self.epsilon)

    def update(self, reward):
        """
        Update Value function depending on reward obtained
        """
        self.r_sum[self.last_action] += reward
        self.action_count[self.last_action] += 1
        self.Q[self.last_action] = self.r_sum[self.last_action]/self.action_count[self.last_action]

    def reset(self):
        # Reinitialise Agent values
        self.Q = np.zeros(self.N)
        self.r_sum = np.zeros(self.N) # Reward Sum for each action
        self.action_count = np.zeros(self.N) # Nb of times each action was taken
        self.last_action = None

class Testbed(object):
    """
    k-arm testbed:
        True values of each action, q*(a), is calculated by sampling a normal dist  with mean 
        zero and unit standard deviation. The actual rewards are then calculated by sampling 
        normal distributions with mean q*(a) and unit deviation
    """
    def __init__(self, k):
        self.k = k

    def reset(self):
        self.action_rewards = np.random.normal(0, 1, self.k) # Normal with mean 0 and unit deviation
    
    def getReward(self, action):
        return np.random.normal(self.action_rewards[action], 1, 1) # Normal with mean q*(a) and unit dev

    def getOptimAction(self):
        return np.argmax(self.action_rewards)

class Enviroment(object):
    """
    Model of the 10 bandit action rewards
    """
    def __init__(self, agents, testbed, iterations, nb_plays):
        self.agents = agents
        self.testbed = testbed
        self.iterations = iterations
        self.plays = nb_plays

    def play(self):

        scores = np.zeros((self.plays, len(self.agents))) # Scores of each agent
        optimals = np.zeros((self.plays, len(self.agents))) # Times the agent took optimal action

        for it in range(self.iterations):
            # Print statement after every 100 iterations
            if (it%100) == 0:
                print("Completed Iterations: ",it)

            # Reset
            self.testbed.reset()
            for agent in self.agents:
                agent.reset()

            for play in range(self.plays):
                agent_ctn = 0
                for agent in self.agents:
                    # Agent chooses an action
                    action_taken = agent.step()

                    # Get real reward of action
                    reward = self.testbed.getReward(action_taken)

                    # Update agent's policy
                    agent.update(reward)

                    scores[play, agent_ctn] += reward

                    # Check if algorithm took optimal action
                    optimal_action = self.testbed.getOptimAction()
                    if action_taken == optimal_action:
                        optimals[play, agent_ctn] += 1

                    agent_ctn += 1

        score_avg = scores/self.iterations
        optimals_avg = optimals/self.iterations

        return score_avg, optimals_avg

def main():

    k = 10
    agents = [Agent(k), Agent(k, 0.1), Agent(k, 0.01)]
    testbed = Testbed(k)
    iterations = 2000
    plays = 1000

    env = Enviroment(agents, testbed, iterations, plays)
    print("Running..")
    score_avg, optimals_avg = env.play()

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