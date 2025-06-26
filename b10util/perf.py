from contextlib import contextmanager
import time
import pprint
from torch.profiler import profile, record_function, ProfilerActivity


class PerfCounterDict(dict):
    def __del__(self):
        pprint.pprint(self)

@contextmanager
def time_block(name):
    with record_function(name):
        if not hasattr(time_block, "perf_counters"):
            time_block.perf_counters = PerfCounterDict()
        
        if not hasattr(time_block, "last_log_time"):
            time_block.last_log_time = time.time()
        
        if time.time() - time_block.last_log_time >= 5:
            pprint.pprint(time_block.perf_counters)
            time_block.last_log_time = time.time()

        start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start
            if name not in time_block.perf_counters:
                time_block.perf_counters[name] = [0, 0]
            time_block.perf_counters[name][0] += 1
            time_block.perf_counters[name][1] += elapsed
    