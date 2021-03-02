import pycurl
from urllib.parse import urlencode
from io import BytesIO
import pandas as pd
from bs4 import BeautifulSoup
import json

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
        'd' : delay}


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


def PartOne(max_clients, max_aps, max_servers, sciper, repeats = 10):
    """
    Try various values for input variables
    """
    
    # Vary the number of clinets
    sim_results = []
    for clients in range(1, max_clients + 1):
        for aps in range(1, max_aps + 1):
            for servers in range(1, max_servers + 1):
                print("Performing repeated simulatinos (%d,%d,%d)"%(clients, aps, servers))
                repeated_sims = [runSimulation(clients, servers, aps, sciper) for r in range(0, repeats)]
                sim_results.append(
                    {'clients' : clients, 
                    'aps': aps,
                    'serveers': servers,
                    'results': repeated_sims}
                )
    return {
        'name' : 'PartOne',
        'configs': {
            'max_clients': max_clients,
            'max_aps': max_aps,
            'max_servers': max_servers,
            'repeats': repeats,
            'sciper' : sciper
        },
        'results':
            sim_results
    }


def PartTwo(max_clinets, sciper, repeats = 10):

    """
        With servers = access points = 1, linearly increase the number of
        clients
    """
    sim_results = []
    for clients in range(1, max_clinets + 1):
        repeated_sims = [runSimulation(clients, 1, 1, sciper) for r in range(0, repeats)]
        sim_results.append(
            {'clients' : clients,
            'results': repeated_sims
            }
        )
    return sim_results

def dumpDict(my_dict, file_name):
    with open(file_name, 'w') as fp:
        pretty_dict = json.dumps(my_dict, indent=4)
        fp.write(pretty_dict)
        
if __name__ == "__main__":

    sciper = 273472
    repeats = 10
    """
    Run part one, repeatedly try out different configurations
    """
    part_one_results = PartOne(4, 4, 4, sciper, repeats)
    dumpDict(part_one_results)
    

    """
    Run part two, keep AP=S=1 and change C

    """
    part_two_results = PartTwo(20, sciper, repeats)
    dumpDict(part_two_results)
    
    
