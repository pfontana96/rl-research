from common.agents import SAAgent, WAAgent, UCB_SAAgent, GAgent
from common.enviroments import Enviroment, StatTestbed
from pathlib import Path

import matplotlib.pyplot as plt 
import numpy as np

def main():
    thetas = np.logspace(-7, 2, num=10, base=2.0) # Parameters value in log2 scale
    scores_avg = np.zeros((len(thetas), 4)) # Average scores (of average scores on n iterations) on m plays
    iterations = 1000
    plays = 1000
    k = 10

    testbed = StatTestbed(k)

    for i in range(len(thetas)):
        agents = [SAAgent(thetas[i], k), WAAgent(0.1, thetas[i], k), UCB_SAAgent(thetas[i], k), GAgent(thetas[i], k)]
        env = Enviroment(agents, testbed, iterations, plays)
        scores, _ = env.run()
        scores_avg[i,:] = np.sum(scores, axis=0)/plays

    path = Path(__file__).resolve() # Path (and file name) to save the image
    legends = ['e-Greedy', 'Greedy with optimistic initialization (alpha: 0.1)', 'UCB', 'Gradient Bandit']
    plt.plot(thetas, scores_avg)
    plt.xscale("log", base=2)
    plt.title("Algorithms comparisson in {}-bandit problem (stationary)".format(k))
    plt.ylabel("Average reward over {} plays ran {} times".format(plays, iterations))
    plt.xlabel(r'Parameter: $\alpha $\epsilon c')
    plt.legend(legends, loc=4, prop={'size':6})    
    plt.savefig(path.with_suffix('.jpg'))

if __name__ == "__main__":
    main()