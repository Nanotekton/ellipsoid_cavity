# Importing the library
import os
import psutil
import numpy as np
from multiprocessing import Process, Queue
import time

def get_usage(interval=4):
    # Calling psutil.cpu_precent() for 4 seconds
    cpu_usage = psutil.cpu_percent(interval)
    
    # Getting all memory using os.popen()
    total_memory, used_memory, free_memory = map(
        int, os.popen('free -t -m').readlines()[-1].split()[1:])
 
    # Memory usage
    memory_usage = (used_memory/total_memory) * 100

    return [cpu_usage, memory_usage]


def monitoring_process(q_in, q_out, interval=4):
    results = []
    while True:
        try:
            command = q_in.get(timeout=1, block=False)
        except:
            command = ''
        if command == 'break':
            break
        results.append(get_usage(interval=interval))
    q_out.put(results)


def run_and_monitor_status(func, args=tuple(), kwargs=dict(), interval=2):
    '''Returns func result and resource info: dict of dicts
    1st level: backgound, process, wall_time
    2nd level: mean, max -> 2-vector of corresponding [cpu, ram] usage'''
    qin, qout = Queue(), Queue()
    monitor = Process(target=monitoring_process, args=(qin, qout), kwargs=dict(interval=interval))
    background = [get_usage(interval) for _ in range(5)]
    
    monitor.start()
    start = time.time()
    func_result = func(*args, **kwargs)
    wall_time = time.time() - start

    qin.put('break')
    monitor_result = qout.get()
    monitor.kill()
    monitor.join()

    stats = {'wall_time':wall_time}
    stats['background'] = {'mean':np.mean(background, axis=0), 'max':np.max(background, axis=0)}
    stats['process'] = {'mean':np.mean(monitor_result, axis=0), 'max':np.max(monitor_result, axis=0)}

    return func_result, stats


