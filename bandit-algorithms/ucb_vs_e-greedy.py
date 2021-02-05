"""
Script to verify advantages of agents using Upper-Confidence-Bounds action-value estimates over
agents using Sample-Average estimates in stationary problems, presented in Sutton 
(Reinforcement Learning 2018) p.36 fig 2.4
"""

from common.enviroments import Enviroment, StatTestbed
from common.agents import SAAgent, UCB_SAAgent
from common.utils import compare
from pathlib import Path

def main():
    k = 10
    c = 2
    epsilon = 0.1
    agents = [SAAgent(epsilon, k), UCB_SAAgent(c, k)]
    testbed = StatTestbed(k)
    iterations = 2000
    plays = 1000

    env = Enviroment(agents, testbed, iterations, plays)
    path = Path(__file__).resolve() # Path (and file name) to save the image

    compare(env, path)

if __name__ == "__main__":
    main()