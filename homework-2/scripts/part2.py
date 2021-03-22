import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib
import statistics


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
        def __init__(self, coord, t: float, v:float) -> None:
            self.t = t
            self.m = coord
            self.v = v

        def set_v(self, v:float) ->None:
            self.v = v

    def __init__(self, num_users, dim_x, dim_y, sciper) -> None:

        self.num_users = num_users
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.sciper = sciper
        self.v_min = 1
        self.v_max = 2
        self.z = [[RandomWaypoint.State(self.randpos(), 0, self.randvelocity())]
                  for user in range(0, self.num_users)]

    def randpos(self) -> Coord:
        return RandomWaypoint.Coord(np.random.uniform(0, self.dim_x),
                                    np.random.uniform(0, self.dim_y))

    def randvelocity(self) -> float:
        return np.random.uniform(self.v_min, self.v_max)

    def simulate(self, t_max):
        start_time = time.time()
        for user_states in self.z:
            current_time = 0
            while (current_time <= t_max): ## This is a slight approximation, we let the user finish his/her last transition
                current_state = user_states[-1]
                current_pos = current_state.m
                current_v = current_state.v
                next_pos = self.randpos()
                next_t = current_state.t + \
                    next_pos.distance(current_pos) / current_v
                next_state = RandomWaypoint.State(next_pos, next_t, self.randvelocity())
                user_states.append(next_state)
                current_time = next_t
        end_time = time.time()
        return end_time - start_time        
    
    def return_user_stats(self) -> (float, float, float):
        stats_list = [len(user_states) for user_states in self.z]
        mean = sum(stats_list)/len(stats_list)
        maximum = max(stats_list)
        minimum = min(stats_list)

        return (mean, maximum, minimum)
        
    def plot_waypoints(self, num_users):

        fig, axs = plt.subplots(2)

        def __plot_user__(states, label, ax, trajectory=False):
            x = [_s.m.x for _s in states]
            y = [_s.m.y for _s in states]
            ax.plot(x, y, marker='x', markersize=0.7,
                    linestyle='-' if trajectory else '', label=label, linewidth=0.5)
            title = 'Trajectory plot' if trajectory else 'Way points'
            ax.set_title(title)

        for ix, user_states in enumerate(self.z[0:num_users]):
            __plot_user__(user_states, 'user ' + str(ix + 1),
                          axs[0], trajectory=True)
            axs[0].legend()
            __plot_user__(user_states, 'user ' + str(ix + 1),
                          axs[1], trajectory=False)
            axs[1].legend()
        fig.set_size_inches((12.80 / 2, 12.80))
        fig.tight_layout()
        fig.savefig('../figures/problem3_' + str(num_users) + '_traj.pdf')
        plt.show()
    
    def plot_event_view(self, num_users):
        fig, axs = plt.subplots(2)
        velocities = list()
        x = list()
        y = list()
        for user_states in self.z[0:num_users]:
            for _s in user_states:
                velocities.append(_s.v)
                x.append(_s.m.x)
                y.append(_s.m.y)
        axs[0].hist(velocities, bins=20, range=(1,2), density=True)
        axs[0].set_xlabel('Velocity (m/s)')
        axs[0].set_ylabel('Probability')
        
        axs[1].hist2d(x=x, y=y, bins=[10,10], range=[[0,1000], [0,1000]], density = True)
        axs[1].set_xlabel('X Coordinate')
        axs[1].set_ylabel('Y Coordinate')
    
        fig.set_size_inches((12.80, 12.80))
        fig.tight_layout()
        fig.savefig('../figures/event_view_' + str(num_users) + '.pdf')
        plt.show()
    
    def get_time_series(self, user_id, t_step, num_steps):
        ## Assumption is that start time is 0
        time = 0
        state_ctr = 0
        time_series = list()
        states = self.z[user_id]
        for i in range(0, num_steps):
            time = time + t_step
            while(states[state_ctr].t < time):
                state_ctr = state_ctr + 1
            state_ctr = state_ctr - 1
            time_fraction = (time - states[state_ctr].t)/(states[state_ctr+1].t - states[state_ctr].t)
            x = states[state_ctr].m.x + (states[state_ctr+1].m.x - states[state_ctr].m.x)* time_fraction
            y = states[state_ctr].m.y + (states[state_ctr+1].m.y - states[state_ctr].m.y)* time_fraction
            v = states[state_ctr].v
            time_series.append(RandomWaypoint.State(RandomWaypoint.Coord(x,y),time,v))
        
        return time_series

    def plot_time_view(self, num_users):
        fig, axs = plt.subplots(2)
        velocities = list()
        x = list()
        y = list()
        for user_id in range(0,num_users):
                time_series = self.get_time_series(user_id,10,8639)
                for _s in time_series:
                    velocities.append(_s.v)
                    x.append(_s.m.x)
                    y.append(_s.m.y)

        axs[0].hist(velocities, bins=20, range=(1,2), density=True)
        axs[0].set_xlabel('Velocity (m/s)')
        axs[0].set_ylabel('Probability')
        
        axs[1].hist2d(x=x, y=y, bins=[10,10], range=[[0,1000], [0,1000]], density = True)
        axs[1].set_xlabel('X Coordinate')
        axs[1].set_ylabel('Y Coordinate')
    
        fig.set_size_inches((12.80, 12.80))
        fig.tight_layout()
        fig.savefig('../figures/time_view_' + str(num_users) + '.pdf')
        plt.show()

    def conf_interval(self, num_users):

        def median_95_conf_interval_indices(num_samples) -> (int,int):
            if(num_samples == 30):
                return (10,21)
            elif (num_samples > 70):
                j = int(0.5*num_samples - 0.98*np.sqrt(num_samples))
                k = int(0.5*num_samples + 1 + 0.98*np.sqrt(num_samples))
                return (j,k)
            else:
                assert(0 and "num_samples not supported")

        event_avg_speed = list()
        time_avg_speed = list()
        for user_id in range(0,num_users):
            event_speeds = [_s.v for _s in self.z[user_id]]
            time_speeds = [_s.v for _s in self.get_time_series(user_id,10,8639)]
            event_avg_speed.append(sum(event_speeds)/len(event_speeds))
            time_avg_speed.append(sum(time_speeds)/len(time_speeds))
        
        event_avg_speed.sort()
        time_avg_speed.sort()
        j, k = median_95_conf_interval_indices(num_users)

        event_mean = sum(event_avg_speed)/len(event_avg_speed)
        event_var = np.var(event_avg_speed)
        time_mean = sum(time_avg_speed)/len(time_avg_speed)
        time_var = np.var(time_avg_speed)
        event_interval = 1.96*np.sqrt(event_var)/np.sqrt(num_users)
        time_interval = 1.96*np.sqrt(time_var)/np.sqrt(num_users)

        print("** Median for %d users ** " %(num_users))
        print("%f %f %f"  %(statistics.median(event_avg_speed), event_avg_speed[j], event_avg_speed[k]))
        print("%f %f %f" %(statistics.median(time_avg_speed), time_avg_speed[j], time_avg_speed[k]))
        print("** Mean for %d users ** " %(num_users))
        print("%f %f %f" %(event_mean, event_mean-event_interval, event_mean+event_interval))
        print("%f %f %f" %(time_mean, time_mean-time_interval, time_mean+time_interval))

    def pred_interval(self, num_users):

        def order_statistic_95_indices(num_samples) -> (int,int):
            if(num_samples == 30):
                return (0,0)
            elif (num_samples >= 60):
                j = int((num_samples+1)*0.025)
                k = int((num_samples+1)*0.975)
                return (j,k)
            else:
                assert(0 and "num_samples not supported")
        
        time_avg_speed = list()
        for user_id in range(0,num_users):
            time_speeds = [_s.v for _s in self.get_time_series(user_id,10,8639)]
            time_avg_speed.append(sum(time_speeds)/len(time_speeds))
        
        time_avg_speed.sort()
        j, k = order_statistic_95_indices(num_users)
        time_mean = sum(time_avg_speed)/len(time_avg_speed)
        time_var = np.var(time_avg_speed)
        time_interval = 1.96*np.sqrt(time_var)

        print("** Prediction interval for %d users ** " %(num_users))
        print("Assuming normal data\n %f - %f" %(time_mean-time_interval, time_mean+time_interval))
        if(num_users > 39):
            print("Assuming order statistic\n %f - %f" %(time_avg_speed[j], time_avg_speed[k]))

if __name__ == "__main__":

    font = {'family': 'sans-serif',
            'sans-serif': ['Helvetica'],
            'size': 18}
    matplotlib.rc('font', **font)

    sim_obj = RandomWaypoint(150, 1000, 1000, 292888)
    sim_time = sim_obj.simulate(86400)
    print(sim_time)
    mean, maximum, minimum = sim_obj.return_user_stats()
    print("Mean waypoints: %f. Max waypoints %f. Min waypoints %f." %(mean, maximum, minimum))
    sim_obj.plot_waypoints(1)
    sim_obj.plot_waypoints(5)
    plt.gray()
    sim_obj.plot_event_view(1)
    sim_obj.plot_event_view(sim_obj.num_users)
    sim_obj.plot_time_view(1)
    # sim_obj.plot_time_view(sim_obj.num_users)
    sim_obj.conf_interval(150)
    sim_obj.conf_interval(30)
    sim_obj.pred_interval(150)
    sim_obj.pred_interval(60)
    sim_obj.pred_interval(30)
