"""
Script to verify advantages of agents using Weigthed-Average action-value estimates over
agents using Sample-Average estimates in non-stationary problems, presented in Sutton 
(Reinforcement Learning 2018) p.33 ex 2.5
"""
from common.enviroments import Enviroment, NStatTestbed
from common.agents import SAAgent, WAAgent
import matplotlib.pyplot as plt

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