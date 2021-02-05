import matplotlib.pyplot as plt

def compare(enviroment, pathname):
    """
    Runs the simulation and saves the results of the agents as a .jpg

    Arguments:
    ----------
    enviroment(common.Enviroment): Enviroment to simulate.
    pathname(pathlib.Path): Path where to save the image.
    """

    print("Running simulation..")
    k = enviroment.testbed.k

    scores_avg, optimals_avg = enviroment.run()

    fig = plt.figure()

    # Graph 1 - average score over time in independant iterations
    ax1 = fig.add_subplot(211)
    ax1.plot(scores_avg)
    ax1.set_title("{k}-Armed TestBed - Average Rewards".format(k=k))
    ax1.set_ylabel('Average Reward')
    ax1.set_xlabel('Plays')
    ax1.legend(enviroment.agents, loc=4, prop={'size':6})

    # Graph 2 - optimal selections over all plays over time in independant iterations
    ax2 = fig.add_subplot(212)
    ax2.plot(optimals_avg * 100)
    ax2.set_title("{k}-Armed TestBed - % Optimal Action".format(k=k))
    ax2.set_ylim(0, 100)
    ax2.set_ylabel('% Optimal Action')
    ax2.set_xlabel('Plays')
    ax2.legend(enviroment.agents, loc=4, prop={'size':6})

    # Save figure
    fig.tight_layout()
    output_file = pathname.with_suffix('.jpg')
    fig.savefig(output_file)
    print('Comparisson saved in :{}'.format(output_file))

