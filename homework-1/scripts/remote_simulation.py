import pycurl
from urllib.parse import urlencode
from io import BytesIO
import pandas as pd
from bs4 import BeautifulSoup
import json
import math


def parseSimRes(res_string):
    soup = BeautifulSoup(res_string, 'html.parser')
    table = soup.find_all('table')[0]

    rows = []
    for child in table.children:
        row = []
        for td in child:
            try:
                row.append(td.text.replace('\n', ''))
            except:
                continue
        if len(row) > 0:
            rows.append(row)

    # It's ugly but it works
    rps = float(rows[2][1])
    aps = float(rows[3][1])
    servers = float(rows[4][1])

    theta = float(rows[6][1])

    pps = float(rows[7][1])
    cps = float(rows[8][1])
    delay = float(rows[9][1])
    return {
        'rps': rps,
        'aps': aps,
        'servers': servers,
        'theta': theta,
        'pps': pps,
        'cps': cps,
        'd': delay}


def runSimulation(clients, servers, apoints, sciper):
    """
    Runs a curl command like this one
    curl -i -X POST -F 'clients=1' -F 'servers=1' -F 'apoints=1' -F 'submit=Run+Simulation' -F 'sciper=123456' https://tcpip.epfl.ch/output.php
    """
    c = pycurl.Curl()

    post_data = {'clients': str(clients), 'servers': str(servers), 'apoints': str(
        apoints), 'submit': 'Run+Simulation', 'sciper': str(sciper)}

    c.setopt(pycurl.URL, 'https://tcpip.epfl.ch/output.php')

    postfields = urlencode(post_data)
    buffer = BytesIO()
    c.setopt(pycurl.POSTFIELDS, postfields)
    c.setopt(pycurl.WRITEFUNCTION, buffer.write)
    c.perform()
    c.close()
    sim_res = buffer.getvalue().decode('UTF-8')
    return parseSimRes(sim_res)


def PartOne(max_clients, max_aps, max_servers, sciper, repeats=10):
    """
    Try various values for input variables
    """

    # Vary the number of clinets
    sim_results = []
    for clients in range(1, max_clients + 1):
        for aps in range(1, max_aps + 1):
            for servers in range(1, max_servers + 1):
                print("Performing repeated simulatinos (%d,%d,%d)" %
                      (clients, aps, servers))
                repeated_sims = [runSimulation(
                    clients, servers, aps, sciper) for r in range(0, repeats)]
                sim_results.append(
                    {'clients': clients,
                     'aps': aps,
                     'servers': servers,
                     'results': repeated_sims}
                )
    return {
        'name': 'PartThree',
        'configs': {
            'max_clients': max_clients,
            'max_aps': max_aps,
            'max_servers': max_servers,
            'repeats': repeats,
            'sciper': sciper
        },
        'results':
            sim_results
    }


def PartTwo(max_clinets, sciper, repeats=10, steps=10):
    """
        With servers = access points = 1, linearly increase the number of
        clients
    """
    sim_results = []

    for clients in range(steps, max_clinets + 1, steps):
        print("Part two tests (%d)" % clients)
        repeated_sims = [runSimulation(clients, 1, 1, sciper)
                         for r in range(0, repeats)]
        sim_results.append(
            {'clients': clients,
             'results': repeated_sims
             }
        )
    return {'name': 'PartTwo',
            'configs':  {
                'max_clients': max_clinets,
                'sciper': sciper,
                'repeats': repeats
            },
            'results': sim_results
            }


def PartThree(max_clients, max_aps, servers, sciper, repeats=10, steps=10):
    """

    """
    sim_results = []
    for aps in range(1, max_aps + 1):
        print("Running part three with %d access points" % aps)

        sim_results.append({
            'aps': aps,
            'results':
            [{
                'clients': clients,
                'results': [runSimulation(
                    clients, servers, aps, sciper) for r in range(0, repeats)]
            } for clients in range(steps, max_clients + 1, steps)]
        })

    return {'name': 'PartTwo',
            'configs':  {
                'max_clinets': max_clients,
                'max_aps': max_aps,
                'sciper': sciper,
                'repeats': repeats
            },
            'results': sim_results
            }


def PartFour(max_clients, max_servers, aps, sciper, repeats=10, steps=10):
    """

    """
    sim_results = []
    for servers in range(1, max_servers + 1):
        print("Running part four with %d servers" % servers)

        sim_results.append({
            'servers': servers,
            'results':
            [{
                'clients': clients,
                'results': [runSimulation(
                    clients, servers, aps, sciper) for r in range(0, repeats)]
            } for clients in range(steps, max_clients + 1, steps)]
        })

    return {'name': 'PartTwo',
            'configs':  {
                'max_clinets': max_clients,
                'max_servers': servers,
                'sciper': sciper,
                'repeats': repeats
            },
            'results': sim_results
            }

def PartFive(max_clinets, sciper, repeats=10, steps=10):
    """
        Linearly increase the number of clients.
        Vary servers and APs based on engineering rule. 
    """
    sim_results = []
    ap_load = 80
    server_load = 250
    for clients in range(steps, max_clinets + 1, steps):
        num_servers = min(math.ceil(clients/server_load),10)
        num_aps = min(math.ceil(clients/ap_load),10)
        print("Part five tests %d clients, %d APs, %d servers" % (clients, num_aps, num_servers))
        repeated_sims = [runSimulation(clients, num_servers, num_aps, sciper)
                         for r in range(0, repeats)]
        sim_results.append(
            {'clients': clients,
             'results': repeated_sims
             }
        )
    return {'name': 'PartTwo',
            'configs':  {
                'max_clients': max_clinets,
                'sciper': sciper,
                'repeats': repeats
            },
            'results': sim_results
            }
  
def dumpDict(my_dict, file_name):
    with open(file_name, 'w') as fp:
        pretty_dict = json.dumps(my_dict, indent=4)
        fp.write(pretty_dict)


if __name__ == "__main__":

    sciper = 273472
    repeats = 10
    steps = 10
    max_clients = 1000
    # """
    # Run part one, repeatedly try out different configurations
    # """
    # part_one_results = PartOne(4, 4, 4, sciper, repeats)
    # dumpDict(part_one_results, '../data/part_one.json', )

    # """
    # Run part two, keep AP=S=1 and change C

    # """
    # part_two_results = PartTwo(max_clients, sciper, repeats, steps)
    # dumpDict(part_two_results, '../data/part_two.json')

    # """
    # Run part three. 1 server. 1-4 APs
    # """
    # part_three_results = PartThree(max_clients, 4, 1, sciper, repeats, steps)
    # dumpDict(part_three_results, '../data/part_three.json')


    # """
    # Run part four (a). 10 servers, 1-10 APs
    # """
    # part_four_results = PartThree(max_clients, 10, 10, sciper, repeats, steps)
    # dumpDict(part_four_results, '../data/part_four.json')

    # """
    # Run part four (b). 10 APs, 1-10 servers.
    # """
    # part_four_results = PartFour(max_clients, 10, 10, sciper, repeats, steps)
    # dumpDict(part_four_results, '../data/part_five.json')

    """
    Run part five, change C and vary AP,S based on engineering rule

    """
    part_two_results = PartFive(max_clients, sciper, repeats, steps)
    dumpDict(part_two_results, '../data/part_six.json')