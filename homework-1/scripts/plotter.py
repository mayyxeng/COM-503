import matplotlib.pyplot as plt
import matplotlib.cm
import numpy as np
import json


def plotPartOne(data_file):

  """
  Plot the data from part one:
  Plot theta, pps, d, cp for each repated points.
  For each configuration of clients, aps, servers a scatter plot
  with unique colors is plotted. So there will be a total 
  of color_range = max_clients * max_aps * max_servers  and 
  color_range * repeats points per plot
  """
  with open(data_file, 'r') as fp:

    data = json.load(fp)

    max_clients = data['configs']['max_clients']
    max_servers = data['configs']['max_servers']
    max_aps = data['configs']['max_aps']
    repeats = data['configs']['repeats']
    fig, axs = plt.subplots(2, 2)
    
    color_range = len(data['results'])

    colors = matplotlib.cm.gist_rainbow(np.linspace(0, 1, color_range))
    

    def __inner_plot__(ax, x, points, color, label):
      _x = [x for p in points]
      ax.scatter(_x, points, color=color, label=label)
      ax.set_ylabel(label)
      
    
    x = 1
    
    xtick_labesls = []
    for repeated_points in data['results']:
      
      clients = repeated_points['clients']
      aps = repeated_points['aps']
      servers = repeated_points['servers']
      xtick_labesls.append("(" + str(clients) + "," + str(aps) + "," + str(servers) + ")")

      theta = [res['theta'] for res in repeated_points['results']]
      pps = [res['pps'] for res in repeated_points['results']]
      cps = [res['cps'] for res in repeated_points['results']]
      delay = [res['d'] for res in repeated_points['results']]

      __inner_plot__(axs[0][0], x, theta, colors[x - 1], r'$\theta$')
      __inner_plot__(axs[0][1], x, pps, colors[x - 1], 'pps')
      __inner_plot__(axs[1][0], x, cps, colors[x - 1], 'cps')
      __inner_plot__(axs[1][1], x, delay, colors[x - 1], 'delay')

      x += 1
    def setTicks(ax):
      ax.set_xticks(range(1, len(xtick_labesls) + 1))
      ax.set_xticklabels(xtick_labesls, rotation=90, fontsize='xx-small')
      # ax.minorticks_on()
      
      
    setTicks(axs[0][0])
    setTicks(axs[1][0])
    setTicks(axs[0][1])
    setTicks(axs[1][1])

    fig.set_size_inches((19.20, 10.80))
    fig.tight_layout()
    # fig.suptitle('Results of ' + str(repeats) +  ' repeated experiments for different values of (C, AP, S)', fontsize='xx-small')
    fig.savefig('../data/part_one.pdf')
    plt.show()


if __name__ == "__main__":

  font = {'family':'sans-serif',
          'sans-serif':['Helvetica'],
          'size' : 18}
  matplotlib.rc('font', **font)

  plotPartOne("../data/part_one.json")
