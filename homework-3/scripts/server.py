from enum import Enum
from numpy.random import exponential
from numpy.random import lognormal
from numpy.random import uniform
from numpy import exp
from bisect import bisect_left
import matplotlib.pyplot as plt
import matplotlib
import scipy.stats as stats_utils
from numpy import abs
import numpy as np
import scipy.stats as st


class Server:

    class JobType(Enum):
        REQ = 1  # poisson dist, arrival of type 1 job
        TYPE1 = 2  # lognormal dist, departure of type1 job = arrival of type2
        TYPE2 = 3  # uniform dist, departure of type2 job

    class Event:
        def __init__(self, job_type, firing_time):
            self.type = job_type
            self.time = firing_time

    class EventScheduler:
        def __init__(self, arrival_rate):
            self.event_list = []
            self.rate = arrival_rate

        def addEvent(self, job_type, prev_firing):

            delta = self.__get_delta__(job_type)
            firing_time = prev_firing + delta
            event = Server.Event(job_type, firing_time)

            self.__insert_sorted__(event)

        def __insert_sorted__(self, event):

            self.event_list.append(event)
            self.event_list.sort(key=lambda ev: ev.time)
            # event_times = [_ev.time for _ev in self.event_list]
            # insertion_index = bisect_left(event_times, event.time)
            # self.event_list.insert(insertion_index, event)

        def popEvent(self):
            next_event = self.event_list[0]
            self.event_list = self.event_list[1:]
            return next_event

        def __get_delta__(self, job_type):

            if job_type == Server.JobType.REQ:
                # Poisson distribution, so exponential random variable
                return exponential(1 / self.rate)
            elif job_type == Server.JobType.TYPE1:
                # lognormal distribution
                return lognormal(1, 1)
            elif job_type == Server.JobType.TYPE2:
                return uniform(0.5, 1)

    class TimedStats:

        def __init__(self):
            self.queue_length = []
            self.time_point = []
            self.requests_arrived = []
            self.requests_served = []
            self.type1_count = []
            self.type2_count = []

    """
      create a server simulator object with the given requests_rate (requests / sec)
    """

    def __init__(self, request_rate, verbose=False):

        self.verbose = verbose
        self.job_queue = []
        self.event_schedule = Server.EventScheduler(request_rate / 1000.)
        self.current_time = 0

        # The first TYPE1 event happens at time 0
        self.event_schedule.addEvent(Server.JobType.REQ, 0)

        self.stats = Server.TimedStats()

        self.requests_served = 0
        self.requests_arrived = 0

        # Per job queue length counters i.e, B(t^{-1}) at every event
        self.job_count = {Server.JobType.TYPE1: 0, Server.JobType.TYPE2: 0}

        # Response time counters
        self.response_counter = {
            Server.JobType.TYPE1: 0, Server.JobType.TYPE2: 0}

        # Number of jobs servred
        self.jobs_served = {
            Server.JobType.TYPE1: 0, Server.JobType.TYPE2: 0
        }

    def __update_stats__(self, new_event):

        self.stats.time_point.append(self.current_time)
        self.stats.queue_length.append(len(self.job_queue))
        self.stats.requests_arrived.append(self.requests_arrived)
        self.stats.requests_served.append(self.requests_served)
        self.stats.type1_count.append(self.job_count[Server.JobType.TYPE1])
        self.stats.type2_count.append(self.job_count[Server.JobType.TYPE2])

    def log(self, msg):
        if self.verbose:
            print("[{time:.3f} ms]:\n\t{msg}".format(
                time=self.current_time, msg=msg))

    def __append_job__(self, job_type):

        self.job_queue.append(job_type)
        self.job_count[job_type] = self.job_count[job_type] + 1

    def __clean_job__(self):
        done_job = self.job_queue[0]
        self.job_queue = self.job_queue[1:]
        self.jobs_served[done_job] = self.jobs_served[done_job] + 1
        self.job_count[done_job] = self.job_count[done_job] - 1

    def __process_job_from_queue__(self, firing_time):
        # pop the job
        new_job = self.job_queue[0]

        # process the job
        self.event_schedule.addEvent(new_job, firing_time)

    def __update_response_time_counter__(self, event_time, job_type):

        self.response_counter[job_type] = \
            self.response_counter[job_type] + \
            (event_time - self.current_time) * \
            self.job_count[job_type]

    def simulate(self, num_req, stationary_only=False):

        stationary_time = 0
        stationary_queue = {
            job: self.job_count[job]
            for job in [Server.JobType.TYPE1, Server.JobType.TYPE2]
        }

        def compute_response_time(job_type):
            return self.response_counter[job_type] / self.jobs_served[job_type]

        def compute_average_served(job_type):
            return self.jobs_served[job_type] / (self.current_time - stationary_time)

        def get_queue_metrics():
            return {
                # note that N = backlogCoutner / T, but backlogCounter  = responseTimeCounter
                job: self.response_counter[job] / self.current_time
                for job in [Server.JobType.TYPE1, Server.JobType.TYPE2]
            }

        stationary_found = False
        stationary_counter = 0

        current_queue_meteric = {
            job: 0 for job in [Server.JobType.TYPE1, Server.JobType.TYPE2]
        }
        while self.requests_arrived <= num_req:

            # get the next event
            current_event = self.event_schedule.popEvent()

            self.__update_stats__(current_event)

            self.log("Processing event {type} with firing time {time:.3f} ms with queue length {q}"
                     .format(type=current_event.type, time=current_event.time, q=len(self.job_queue)))

            # update the response time counter
            self.__update_response_time_counter__(
                current_event.time, Server.JobType.TYPE1)
            self.__update_response_time_counter__(
                current_event.time, Server.JobType.TYPE2)

            if current_event.type == Server.JobType.REQ:  # arrival of a new request
                # the current event is an arrival of a new request, append the new request
                # to the server queue and create and event for the next request

                self.requests_arrived = self.requests_arrived + 1
                self.__append_job__(Server.JobType.TYPE1)
                # if this is the first request, start processing this request
                if len(self.job_queue) == 1:
                    self.event_schedule.addEvent(
                        Server.JobType.TYPE1, current_event.time)

                # create the next TYPE1 event
                self.event_schedule.addEvent(
                    Server.JobType.REQ, current_event.time)

            elif current_event.type == Server.JobType.TYPE1:  # departure of TYPE1 job
                # the current event indicates that some job of TYPE1 is finished and
                # a job of TYPE2 needs to be created which has lognormal distribution

                # clean the currently finished job
                self.__clean_job__()

                # append job TYPE2 that follow TYPE1
                self.__append_job__(Server.JobType.TYPE2)

                # picka job from the queue and start processing
                self.__process_job_from_queue__(current_event.time)

            elif current_event.type == Server.JobType.TYPE2:
                # A TYPE2 job is finished, a full request has been served
                self.requests_served = self.requests_served + 1

                # clean the currenlty finished job
                self.__clean_job__()

                # pick a job from the queue and start processing if there are some
                if len(self.job_queue) > 0:
                    self.__process_job_from_queue__(current_event.time)

            else:
                raise RuntimeError("Invalid event type!")

            self.current_time = current_event.time

            new_queue_metrics = get_queue_metrics()

            def metrics_stationary(old, new) -> bool:
                if old[Server.JobType.TYPE1] == 0 or old[Server.JobType.TYPE2] == 0:
                    return False
                err = {
                    job: abs(old[job] - new[job]) / old[job]
                    for job in [Server.JobType.TYPE1, Server.JobType.TYPE2]
                }
                self.log("Queue metric erros: TYPE1 {:.4f}, TYPE2 {:.4f}"
                         .format(
                             err[Server.JobType.TYPE1], err[Server.JobType.TYPE2]
                         ))
                return (err[Server.JobType.TYPE1] <= 0.05 and err[Server.JobType.TYPE2] <= 0.05)
            # Check whether performance metrics are stationary
            if metrics_stationary(current_queue_meteric, new_queue_metrics):
                stationary_counter += 1

            if stationary_counter == 10 and stationary_found == False:
                stationary_found = True
                stationary_time = current_event.time
                stationary_queue = {
                    job: self.response_counter[job]
                    for job in [Server.JobType.TYPE1, Server.JobType.TYPE2]
                }
                self.log("Found stationary region")

            current_queue_meteric = get_queue_metrics()

        # Simulation is done, return the average statistics
        job_queue_metrics = {}
        response_time = {}
        if stationary_found == True and stationary_only == True:
            self.log("Stationary region starts at {:.3f}".format(
                stationary_time))
            job_queue_metrics = {
                job: (self.response_counter[job] - stationary_queue[job]
                      ) / (self.current_time - stationary_time)
                for job in [Server.JobType.TYPE1, Server.JobType.TYPE2]
            }
        else:
            self.log("Did not find the stationary region!")
            job_queue_metrics = {
                job: self.response_counter[job] / self.current_time
                for job in [Server.JobType.TYPE1, Server.JobType.TYPE2]
            }
        job_avg = {
            job: {
                "response_time":
                    compute_response_time(job),
                "num_served":
                    self.jobs_served[job],
                "served_per_ms":
                    compute_average_served(job),
                "average_queue":
                    job_queue_metrics[job]
            } for job in [Server.JobType.TYPE1, Server.JobType.TYPE2]
        }

        return {
            "time": self.current_time,
            "jobs": job_avg,
            'stationary': stationary_time if stationary_found == True and stationary_only == True else 0
        }

    def plotRequests(self, file_name):

        fig, ax = plt.subplots()

        ax.plot(self.stats.time_point, self.stats.requests_arrived,
                label='requests arrived')
        ax.plot(self.stats.time_point, self.stats.requests_served,
                label='requests served')
        ax.set_ylabel('Number of requests (arrived/served)')
        ax.set_xlabel('Time (ms)')

        ax.grid(True)
        ax.legend()

        fig.set_size_inches((12.80, 7.68))
        fig.tight_layout()
        plt.show()
        fig.savefig('../figures/' + file_name)

    

    def plotJobQueue(self, title, file_name):

        fig, ax = plt.subplots()

        ax.plot(self.stats.time_point, self.stats.type1_count, label="type 1")
        ax.plot(self.stats.time_point, self.stats.type2_count, label="type 2")
        ax.plot(self.stats.time_point,
                self.stats.queue_length, label="queue length")
        ax.set_ylabel("Number of jobs in queue")
        ax.set_xlabel("Time (ms)")
        ax.grid(True)
        ax.legend()
        ax.set_title(title)

        fig.set_size_inches((12.80, 7.68))
        fig.tight_layout()
        plt.show()
        fig.savefig('../figures/' + file_name)


def printSimRes(res):
    print("""
Total time : {time:.3f} ms,
TYPE1:
    response time:  {t1_rsp: .3f} ms
    served       :  {t1_srvd: .3f} 
    service rate :  {t1_rate: .3f} jobs/ms
    N            :  {t1_q: .3f}
TYPE2:
    response time:  {t2_rsp: .3f} ms
    served       :  {t2_srvd: .3f} 
    service rate :  {t2_rate: .3f} jobs/ms
    N            :  {t2_q: .3f}
""".format(
        time=res['time'],
        t1_rsp=res['jobs'][Server.JobType.TYPE1]['response_time'],
        t1_srvd=res['jobs'][Server.JobType.TYPE1]['num_served'],
        t1_rate=res['jobs'][Server.JobType.TYPE1]['served_per_ms'],
        t1_q=res['jobs'][Server.JobType.TYPE1]['average_queue'],
        t2_rsp=res['jobs'][Server.JobType.TYPE2]['response_time'],
        t2_srvd=res['jobs'][Server.JobType.TYPE2]['num_served'],
        t2_rate=res['jobs'][Server.JobType.TYPE2]['served_per_ms'],
        t2_q=res['jobs'][Server.JobType.TYPE2]['average_queue']

    ))


def Part1():

    rps = 100
    max_req = 10000
    server = Server(rps)
    res = server.simulate(max_req)
    printSimRes(res)
    server.plotRequests('part1_requests.pdf')
    server.plotJobQueue('Queue utilization', 'part1_queues.pdf')


def Part2():
    max_req = 100000

    explanation = \
        """

To find an analytical bound for the stability of the system, we perform a
bottleneck analysis. S1 = exp(1 + 1/2) is the average service time for TYPE1
jobs and S2 = (1 + 0.5) / 2 is the average service time fot TYPE2 jobs.
From the bottleneck analysis, we know that the waiting time for TYPE1 and TYPE2
jobs are non-zero. Furthermore, the serivce time for the whole request can be
modeled as the S1 + W2 + S2, where W2 is the average waiting time for a TYPE2
job.

We have:
W2 >= 0 
lambda * (S1 + S2 + W2) < 1 (for stabitlity)
lambda * S1 < 1
lambda * S2 < 1

Which gives us:
lambda < min(1 / S1, 1 / S2, 1/(S1 + S2)) = 191.20
    """.format()

    print(explanation)

    lambdas = [50, 100, 150, 180, 190, 191, 192, 200, 250]
    for rate in lambdas:
        # analytical_model = MGI1Model(rate / 1000.)
        q_util = rate / 1000. * (exp(3/2) + 0.75)
        server = Server(rate)
        res = server.simulate(max_req * 10)
        print("With lambda = " + str(rate) + " requests / second")
        printSimRes(res)
        server.plotJobQueue(r"$\lambda = {}, \lambda  S = {:.2f}$".format(rate, q_util),
                            "part2_queues_{}.pdf".format(rate))


def Part3():

    rates = [100, 180]
    repeats = 30 + 30
    max_req = 10000

    def analysis(rate, find_stationary):

        results = []
        for i in range(0, repeats):
            server = Server(rate, verbose=False)

            res = server.simulate(max_req, stationary_only=find_stationary)
            # print("Performed simulation with lambda = {:.3f} and stationary = {:.3f}".format(
            #     rate, res['stationary']))

            results.append(res)

        

        def mean_cf_95(data):

            s_n = st.sem(data)
            m_n = np.mean(data)
            eta = 1.96
            interval = eta * np.sqrt(s_n / len(data))
            return (m_n, interval)

        def median_cf_95(data):

            num_samples = len(data)
            if num_samples < 30:
                raise RuntimeError("Not enough data for median cf!")
            else:
                j = int(np.floor(0.5*num_samples - 0.98 * np.sqrt(num_samples)))
                k = int(np.ceil(0.5*num_samples + 0.98 * np.sqrt(num_samples)) + 1)
                med = np.median(data)
                ordered = np.sort(data)
                return (ordered[j], med, ordered[k])

        def print_cf(data, metric):
            data_mean, mean_cf = mean_cf_95(data)
            med_l, data_med, med_h = median_cf_95(data)
            print(metric + ":")
            print("\tMean: {:.5f} +/- {:.5f}".format(data_mean, mean_cf))
            print("\tMedian: {:.5f}, {:.5f}, {:.5f}".format(
                med_l, data_med, med_h
            ))

        type1_N = [r['jobs'][Server.JobType.TYPE1]['average_queue']
                   for r in results]
        type1_R = [r['jobs'][Server.JobType.TYPE1]['response_time']
                   for r in results]
        type1_l = [r['jobs'][Server.JobType.TYPE1]['served_per_ms']
                   for r in results]
        type2_N = [r['jobs'][Server.JobType.TYPE2]['average_queue']
                   for r in results]
        type2_R = [r['jobs'][Server.JobType.TYPE2]['response_time']
                   for r in results]
        type2_l = [r['jobs'][Server.JobType.TYPE2]['served_per_ms']
                   for r in results]
        print("TYPE1 job:")
        print_cf(type1_N, "N")
        print_cf(type1_R, "R")
        print_cf(type1_l, "served_per_ms")
        print("TYPE2 job:")
        print_cf(type2_N, "N")
        print_cf(type2_R, "R")
        print_cf(type2_l, "served_per_ms")

        return (type1_N, type2_N)


    for rate in rates:
        print("LAMBDA = {:.3f}".format(rate))
        print("-----NON-STATIONARY------")
        no_type1_N, no_type2_N = analysis(rate, False)

        print("=========================")
        print("-------STATIONARY--------")
        type1_N, type2_N = analysis(rate, True)
        print("=========================")

        fig, axs = plt.subplots(1, 2)
        
        axs[0].boxplot([no_type1_N, type1_N], labels=['non-stationary', 'stationary'], meanline=True)
        axs[1].boxplot([no_type2_N, type2_N], labels=['non-stationary', 'stationary'], meanline=True)
        axs[0].set_title(r'$\lambda = {:3.0f}$, type 1'.format(rate))
        axs[1].set_title(r'$\lambda = {:3.0f}$, type 2'.format(rate))

        fig.set_size_inches((12.80, 7.68))
        fig.tight_layout()
        plt.show()
        fig.savefig('../figures/part3_' + str(rate) + ".pdf")




def Part4():

    def eval_p0(n):
        return np.log(2 / 0.1) * 1/n
    
    fig, ax = plt.subplots()

    x = [i + 1 for i in range(0, 350)]
    y = [eval_p0(_x) for _x in x]

    ax.plot(x, y)
    ax.set_yscale('log')
    ax.hlines(0.01, 0, 351)
    ax.grid(True)
    plt.show()

if __name__ == "__main__":

    font = {'family': 'sans-serif',
            'sans-serif': ['Helvetica'],
            'size': 18}
    matplotlib.rc('font', **font)
    # Part1()
    # Part2()
    Part3()
    
