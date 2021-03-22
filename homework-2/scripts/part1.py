import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import t as student  # student distribution
from scipy.stats import norm  # normal dist
from scipy.stats import beta  # beta dist
from scipy.stats import expon
import statsmodels.api as sm
from scipy.stats import probplot

def problem1():

    def normal_hist(num_samples, ax):

        label = r'$n = $' + str(num_samples)
        samples = np.random.normal(loc=0, scale=1., size=num_samples)
        count, bins, ignored = ax.hist(samples, 30, density=True,
                                       label=label)

        ax.set_title(label)

    fig, axs = plt.subplots(3, 3)
    for i, j in np.ndindex((3, 3)):
        num_samples = int((2**(i * 3 + j)) * 10)
        normal_hist(num_samples, axs[i][j])

    fig.set_size_inches((12.80, 12.80))
    fig.tight_layout()
    fig.savefig('../figures/problem1.pdf')
    plt.show()


def problem2():

    samples_store = {
        'Normal':
            [
                {'params': (0, 0.5)},
                {'params': (0, 1)},
                {'params': (1, 1)},
                {'params': (0, 2)}
            ],
        'Student': [
                {'params': 1},
                {'params': 6},
        ],
        'Exponential': [
                {'params': 0.25},
                {'params': 4}
        ],
        'Beta': [
                {'params': (1, 1)},
                {'params': (4, 2)},
                {'params': (2, 4)},
                {'params': (2, 2)}
        ]
    }
    

    def plot_student_dist(nu, ax, num_points=1000):

        # values on the x-axis are from 0.01 to 0.99 percentile
        x = np.linspace(student.ppf(0.01, nu),
                        student.ppf(0.99, nu), num_points)
        label = r'$t(' + str(nu) + ')$'
        ax.plot(x, student.pdf(x, nu), label=label, linewidth=4.0, alpha=0.8)

    def plot_normal_dist(mu, sigma, ax, num_points=1000):

        x = np.linspace(norm.ppf(0.01, mu, sigma), norm.ppf(0.99, mu, sigma),
                        num_points)
        label = r'$N(' + str(mu) + ', ' + str(sigma) + ')$'
        ax.plot(x, norm.pdf(x, mu, sigma),
                label=label, linewidth=4.0, alpha=0.8)

    def plot_beta_dist(a, b, ax, num_points=1000):
        x = np.linspace(beta.ppf(0.01, a, b), beta.ppf(0.99, a, b), num_points)
        label = r'$\beta$(' + str(a) + ', ' + str(b) + ')'
        ax.plot(x, beta.pdf(x, a, b), label=label, linewidth=4.0, alpha=0.8)
        ax.set_title(label)

    def plot_expon_dist(lambda__, ax, num_points=1000):
        x = np.linspace(expon.ppf(0.01, lambda__),
                        expon.ppf(0.99, lambda__), num_points)
        label = r'$Exp(' + str(lambda__) + ')$'
        ax.plot(x, expon.pdf(x, lambda__),
                label=label, linewidth=4.0, alpha=0.8)
        ax.set_title(label)

    def plot_dists():

        fig, axs = plt.subplots(2, 2)

        for cfg in samples_store['Normal']:
          mu, sigma = cfg['params']
          plot_normal_dist(mu, sigma, axs[0][0])

        axs[0][0].set_title('Normal')
        axs[0][0].grid()
        axs[0][0].legend()

        for cfg in samples_store['Student']:
          nu = cfg['params']
          plot_student_dist(nu, axs[0][1])
    
        axs[0][1].set_title('Student')
        axs[0][1].grid()
        axs[0][1].legend()

        for cfg in samples_store['Exponential']:
          l =  cfg['params']
          plot_expon_dist(l, axs[1][0])
      
        axs[1][0].set_title('Exponential')
        axs[1][0].grid()
        axs[1][0].legend()

        for cfg in samples_store['Beta']:
          a, b = cfg['params']
          plot_beta_dist(a, b, axs[1][1])

        axs[1][1].set_title('Beta')
        axs[1][1].grid()
        axs[1][1].legend()

        fig.set_size_inches((12.80, 12.80))
        fig.tight_layout()
        fig.savefig('../figures/problem2_dists.pdf')
        plt.show()

    def plot_qq():
      
      fig, axs = plt.subplots(len(samples_store['Normal']))
      for i, cfg in enumerate(samples_store['Normal']):
        mu, sigma = cfg['params']
        cfg['samples'] = np.random.normal(mu, sigma, 1500)
        probplot(cfg['samples'], plot=axs[i])
        axs[i].set_title(r'$N(' + str(mu) + ', ' + str(sigma) + r')$')
      fig.set_size_inches((12.80, 12.80))
      fig.tight_layout()
      fig.savefig('../figures/problem1_normal_qqplot.pdf')
      plt.show()


      fig, axs = plt.subplots(len(samples_store['Student']))
      for i, cfg in enumerate(samples_store['Student']):
        nu = cfg['params']
        cfg['samples'] = np.random.standard_t(nu, 1500)
        probplot(cfg['samples'], plot=axs[i])
        axs[i].set_title(r'$Student(' + str(nu) + r')$')
      fig.set_size_inches((12.80, 12.80))
      fig.tight_layout()
      fig.savefig('../figures/problem1_student_qqplot.pdf')
      plt.show()

      fig, axs = plt.subplots(len(samples_store['Exponential']))
      for i, cfg in enumerate(samples_store['Exponential']):
        l =  cfg['params']
        cfg['samples'] = np.random.exponential(l, 1500)
        probplot(cfg['samples'], plot=axs[i])
        axs[i].set_title(r'$Exponential(' + str(l) + r')$')
      fig.set_size_inches((12.80, 12.80))
      fig.tight_layout()
      fig.savefig('../figures/problem1_exponential_qqplot.pdf')
      plt.show()
    
      fig, axs = plt.subplots(len(samples_store['Beta']))
      for i, cfg in enumerate(samples_store['Beta']):
        a, b = cfg['params']
        cfg['samples'] = np.random.beta(a, b, 1500)
        probplot(cfg['samples'], plot=axs[i])
        axs[i].set_title(r'$\beta(' + str(a) + ', ' + str(b) + r')$')
      fig.set_size_inches((12.80, 12.80))
      fig.tight_layout()
      fig.savefig('../figures/problem1_beta_qqplot.pdf')
      plt.show()


        


    plot_dists()
    plot_qq()


if __name__ == "__main__":
    font = {'family': 'sans-serif',
            'sans-serif': ['Helvetica'],
            'size': 18}
    matplotlib.rc('font', **font)
    problem1()
    problem2()
