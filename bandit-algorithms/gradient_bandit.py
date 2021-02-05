"""
Script to shown the performance decay in Gradient-Bandit algorithm when the baseline
term is ommited (which is mathematically valid) presented in Sutton (Reinforcement Learning 
2018, p.38 fig 2.5)
"""

from common.enviroments import Enviroment, StatTestbed
from common.agents import GAgent
from common.utils import compare
from pathlib import Path

def main():
    k = 10
    agents = [GAgent(0.1, k), GAgent(0.1, k, baseline=False), GAgent(0.4, k), GAgent(0.4, k, baseline=False)]
    testbed = StatTestbed(k, mean=4)
    iterations = 2000
    plays = 1000

    env = Enviroment(agents, testbed, iterations, plays)
    path = Path(__file__).resolve() # Path (and file name) to save the image

    compare(env, path)

if __name__ == "__main__":
    main()