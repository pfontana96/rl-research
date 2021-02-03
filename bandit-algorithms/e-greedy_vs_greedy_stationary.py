"""
Script to verify advantages of e-Greedy agents over Greedy agents, both using sample-average
action-value estimates in stationary problems, presented in Sutton (Reinforcement Learning 
2018) p.29
"""
from common.agents import SAAgent
from common.enviroments import Enviroment, StatTestbed
import matplotlib.pyplot as plt

def main():

    k = 10
    agents = [SAAgent(k), SAAgent(k, 0.1), SAAgent(k, 0.01)]
    testbed = StatTestbed(k)
    iterations = 2000
    plays = 1000

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