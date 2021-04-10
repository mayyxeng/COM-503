from enum import Enum
from numpy.random import exponential
from numpy.random import lognormal
from numpy.random import uniform
from bisect import bisect_left
import matplotlib.pyplot as plt
import matplotlib


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

            print("Creating an event {} at time {}".format(job_type, firing_time))
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

    def __init__(self, request_rate):

        self.job_queue = []
        self.event_schedule = Server.EventScheduler(request_rate / 1000.)
        self.current_time = 0

        # The first TYPE1 event happens at time 0
        self.event_schedule.addEvent(Server.JobType.REQ, 0)

        self.stats = Server.TimedStats()

        self.requests_served = 0
        self.requests_arrived = 0
        self.job_count = {Server.JobType.TYPE1: 0, Server.JobType.TYPE2: 0}

    def __update_stats__(self, new_event):

        self.stats.time_point.append(self.current_time)
        self.stats.queue_length.append(len(self.job_queue))
        self.stats.requests_arrived.append(self.requests_arrived)
        self.stats.requests_served.append(self.requests_served)
        self.stats.type1_count.append(self.job_count[Server.JobType.TYPE1])
        self.stats.type2_count.append(self.job_count[Server.JobType.TYPE2])

    def log(self, msg):
        print("[{time:.3f} ms]:\n\t{msg}".format(
            time=self.current_time, msg=msg))

    def __append_job__(self, job_type):

        self.job_queue.append(job_type)
        self.job_count[job_type] = self.job_count[job_type] + 1

    def __clean_job__(self):
        done_job = self.job_queue[0]
        self.job_queue = self.job_queue[1:]
        self.job_count[done_job] = self.job_count[done_job] - 1

    def __process_job_from_queue__(self, firing_time):
        # pop the job
        new_job = self.job_queue[0]

        # process the job
        self.event_schedule.addEvent(new_job, firing_time)

    def simulate(self, num_req):

        while self.requests_arrived <= num_req:

            # get the next event
            current_event = self.event_schedule.popEvent()

            self.__update_stats__(current_event)

            self.log("Processing event {type} with firing time {time:.3f} ms with queue length {q}"
                     .format(type=current_event.type, time=current_event.time, q=len(self.job_queue)))
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

    def plotRequests(self):

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
        fig.savefig('../figures/req_time.pdf')

    def plotJobQueue(self):

        fig, ax = plt.subplots()

        ax.plot(self.stats.time_point, self.stats.type1_count, label="type 1")
        ax.plot(self.stats.time_point, self.stats.type2_count, label="type 2")
        ax.plot(self.stats.time_point,
                self.stats.queue_length, label="queue length")
        ax.set_ylabel("Number of jobs in queue")
        ax.set_xlabel("Time (ms)")
        ax.grid(True)
        ax.legend()

        fig.set_size_inches((12.80, 7.68))
        fig.tight_layout()
        plt.show()
        fig.savefig('../figures/typed_job_queue.pdf')


if __name__ == "__main__":

    font = {'family': 'sans-serif',
            'sans-serif': ['Helvetica'],
            'size': 18}
    matplotlib.rc('font', **font)

    rps = 100
    max_req = 100
    server = Server(rps)
    server.simulate(max_req)

    server.plotRequests()
    server.plotJobQueue()
