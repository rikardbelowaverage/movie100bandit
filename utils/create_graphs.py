import matplotlib.pyplot as plt
import numpy as np

def regret_plot(timesteps,regrets,result_path):
    ts = range(0,timesteps)
    plt.plot(ts,regrets,'.',linestyle='')
    plt.savefig(result_path+'/instant_regret.png')
    plt.clf()

    plt.plot(ts,np.cumsum(regrets))
    plt.savefig(result_path+'/cum_regret.png')