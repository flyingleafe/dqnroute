import heapq as hq

class EventQueue:
    """A class to maintain incoming and outgoing event queues"""

    def __init__(self):
        self._heap = []

    def push(self, event):
        hq.heappush(self._heap, event)

    def pop(self):
        return hq.heappop(self._heap)

    def peek(self):
        return self._heap[0]

    def earlier_than(self, time):
        res = []
        self._earlier_than(time, 0, res)
        return res

    def _earlier_than(self, time, k, es):
        if k >= len(self._heap):
            return
        if self._heap[k].time <= time:
            es.append(self._heap[k])
            self._earlier_than(time, 2*k + 1, es)
            self._earlier_than(time, 2*k + 2, es)
