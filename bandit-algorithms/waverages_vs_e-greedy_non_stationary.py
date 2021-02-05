"""
Script to verify advantages of agents using Weigthed-Average action-value estimates over
agents using Sample-Average estimates in non-stationary problems, presented in Sutton 
(Reinforcement Learning 2018) p.33 ex 2.5
"""
from common.enviroments import Enviroment, NStatTestbed
from common.agents import SAAgent, WAAgent
from common.utils import compare
from pathlib import Path

def main():
    k = 10
    epsilon = 0.1
    alpha = 0.1
    agents = [SAAgent(epsilon, k), WAAgent(alpha, epsilon, k)]
    testbed = NStatTestbed(k)
    iterations = 2000
    plays = 10000

    env = Enviroment(agents, testbed, iterations, plays)
    path = Path(__file__).resolve() # Path (and file name) to save the image

    compare(env, path)

if __name__ == "__main__":
    main()