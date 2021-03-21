import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib


class RandomWaypoint:

    class Coord:
        def __init__(self, x, y) -> None:
            self.x = x
            self.y = y

        def distance(self, that) -> float:
            return np.sqrt(
                (self.x - that.x) ** 2 +
                (self.y - that.y) ** 2
            )

    class State:
        def __init__(self, coord, t: float) -> None:
            self.t = t
            self.m = coord

    def __init__(self, num_users, dim_x, dim_y, sciper) -> None:

        self.num_users = num_users
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.sciper = sciper
        self.v_min = 1
        self.v_max = 2
        self.z = [[RandomWaypoint.State(self.randpos(), 0)]
                  for user in range(0, self.num_users)]

    def randpos(self) -> Coord:
        return RandomWaypoint.Coord(np.random.uniform(0, self.dim_x),
                                    np.random.uniform(0, self.dim_y))

    def randvelocity(self) -> float:
        return np.random.uniform(self.v_min, self.v_max)

    def simulate(self, t_max):
        start_time = time.time()
        for user_states in self.z:
            for t in range(0, t_max):
                current_state = user_states[-1]
                current_pos = current_state.m
                next_pos = self.randpos()
                next_t = current_state.t + \
                    next_pos.distance(current_pos) / self.randvelocity()
                next_state = RandomWaypoint.State(next_pos, next_t)
                user_states.append(next_state)

            waypoints = [[_s.x, _s.y] for _s in user_states]
        end_time = time.time()

        
        return end_time - start_time

    def plot_waypoints(self):

        fig, axs = plt.subplots(2)

        def __plot_user__(states, label, ax, trajectory=False):
            x = [_s.m.x for _s in states]
            y = [_s.m.y for _s in states]
            ax.plot(x, y, marker='x', markersize=0.7,
                    linestyle='-' if trajectory else '', label=label, linewidth=0.5)
            title = 'Trajectory plot' if trajectory else 'Way points'
            ax.set_title(title)

        for ix, user_states in enumerate(self.z):
            __plot_user__(user_states, 'user ' + str(ix + 1),
                          axs[0], trajectory=True)
            axs[0].legend()
            __plot_user__(user_states, 'user ' + str(ix + 1),
                          axs[1], trajectory=False)
            axs[1].legend()
        fig.set_size_inches((12.80 / 2, 12.80))
        fig.tight_layout()
        fig.savefig('../figures/problem3_' + str(self.num_users) + '_traj.pdf')
        plt.show()


if __name__ == "__main__":

    font = {'family': 'sans-serif',
            'sans-serif': ['Helvetica'],
            'size': 18}
    matplotlib.rc('font', **font)

    sim_obj = RandomWaypoint(1, 1000, 1000, 292888)
    sim_time = sim_obj.simulate(8640)
    print(sim_time)
    sim_obj.plot_waypoints()
