"""
Script to verify advantages of agents using Upper-Confidence-Bounds action-value estimates over
agents using Sample-Average estimates in stationary problems, presented in Sutton 
(Reinforcement Learning 2018) p.36 fig 2.4
"""

from common.enviroments import Enviroment, StatTestbed
from common.agents import SAAgent, UCB_SAAgent
import matplotlib.pyplot as plt

def main():
    k = 10
    c = 2
    epsilon = 0.1
    agents = [SAAgent(k, epsilon), UCB_SAAgent(c, k, 0)]
    testbed = StatTestbed(k)
    iterations = 2000
    plays = 1000

    env = Enviroment(agents, testbed, iterations, plays)
    print("Running..")
    score_avg, optimals_avg = env.run()

    #Graph 1 - average score over time in independant iterations
    plt.title("{}-Armed TestBed - Average Rewards".format(k))
    plt.plot(score_avg)
    plt.ylabel('Average Reward')
    plt.xlabel('Plays')
    plt.legend(agents, loc=4)
    plt.show()

    #Graph 2 - optimal selections over all plays over time in independant iterations
    plt.title("{}-Armed TestBed - % Optimal Action".format(k))
    plt.plot(optimals_avg * 100)
    plt.ylim(0, 100)
    plt.ylabel('% Optimal Action')
    plt.xlabel('Plays')
    plt.legend(agents, loc=4)
    plt.show()

if __name__ == "__main__":
    main()