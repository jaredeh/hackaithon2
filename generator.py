#! /usr/bin/env python3

import time
from datetime import timedelta, datetime
import random
import json

config = {
    'size': 16*1024,
    'filetype': 6,
    'bucket': 1,
    'ppi_check': 1,
    'source': 10,
    'creation_date': [datetime.now()-timedelta(days=365), datetime.now()-timedelta(days=1)],
    'permission': 4,
    'migrations': {
        'count': 4,
        'bucket': 1,
        'timestamp': [datetime.now()-timedelta(days=3), datetime.now()-timedelta(hours=1)],
    },
    'accesses': {
        'count': 10,
        'rw': 1,
        'timestamp': [datetime.now()-timedelta(days=3), datetime.now()],
        'requestor': 10,
        'majic_requestor': 7,
        'lat': 1000000
    },
}

def get_random_ints(stat, keys, config, do_majic=False):
    majic = ''
    skeys = []
    for key in keys:
        a = key.split('majic_')
        if len(a) > 1:
            majic = a[1]
            majic_value = config[key]
        else:
            skeys.append(key)
    for key in skeys:
        value = random.randint(0, config[key])
        if majic != '':
            while value == majic_value:
                value = random.randint(0, config[key])
            if do_majic:
                value = majic_value
        stat[key] = value

def get_random_timestamp(stat, key, config):
    # this is a date that needs to be randomized in the range
    a = int(config[key][0].timestamp())
    b = int(config[key][1].timestamp())
    #print(f"a={a} b={b}")
    stat[key] = random.randint(a, b)

def get_random_timestamps(stat, aname, keys, config, once=False):
    ts_start = int(config[aname]['timestamp'][0].timestamp())
    ts_end = int(config[aname]['timestamp'][1].timestamp())
    count = random.randint(1, config[aname]['count'])
    for i in range(count):
        if i == 0:
            stat[aname][i]['timestamp'] = random.randint(ts_start, ts_start+3600)
        else:
            ts_last = int(stat[aname][i-1]['timestamp'])
            if ts_last == ts_end:
                return
            stat[aname][i]['timestamp'] = random.randint(ts_last, ts_end)
        ks = list(config[aname].keys())
        ks.remove('count')
        ks.remove('timestamp')
        get_random_ints(stat[aname][i], ks, config[aname], do_majic=True)
        if once:
            stat
            return

def get_stat(special=False):
    stat = {}
    get_random_ints(stat, ['size', 'filetype', 'bucket', 'ppi_check','source', 'permission'], config)
    get_random_timestamp(stat, 'creation_date', config)
    stat['migrations'] = []
    for i in range(config['migrations']['count']):
        stat['migrations'].append({'timestamp': 0, 'bucket': 0})
    stat['accesses'] = []
    for i in range(config['accesses']['count']):
        stat['accesses'].append({'timestamp': 0, 'rw': 0,'requestor': 0, 'lat': 0})
    get_random_timestamps(stat, 'migrations', 'timestamp', config)
    if not special:
        get_random_timestamps(stat, 'accesses', 'timestamp', config)
    else:
        get_random_timestamps(stat, 'accesses', 'timestamp', config, once=True)
    return stat

stats = []
for i in range(100):
    stats.append(get_stat())
for i in range(10):
    stats.append(get_stat(special=True))

print(json.dumps(stats, indent=4))
