"""
Script to verify advantages of e-Greedy agents over Greedy agents, both using sample-average
action-value estimates in stationary problems, presented in Sutton (Reinforcement Learning 
2018) p.29
"""
from common.agents import SAAgent
from common.enviroments import Enviroment, StatTestbed
from common.utils import compare
from pathlib import Path

def main():

    k = 10
    agents = [SAAgent(0, k), SAAgent(0.1, k), SAAgent(0.01, k)]
    testbed = StatTestbed(k)
    iterations = 2000
    plays = 1000

    env = Enviroment(agents, testbed, iterations, plays)
    path = Path(__file__).resolve() # Path (and file name) to save the image

    compare(env, path)
    
if __name__ == "__main__":
    main()