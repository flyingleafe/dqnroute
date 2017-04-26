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

    def change_time(self, event, new_time):
        j = -1
        for (i, e) in enumerate(self._heap):
            if e == event:
                j = i
                break
        if j == -1:
            raise Exception('Event is not present in the queue!')
        self._heap[j].time = new_time
        _siftup(self._heap, j)

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

def _siftdown(heap, startpos, pos):
    newitem = heap[pos]
    # Follow the path to the root, moving parents down until finding a place
    # newitem fits.
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        parent = heap[parentpos]
        if newitem < parent:
            heap[pos] = parent
            pos = parentpos
            continue
        break
    heap[pos] = newitem

def _siftup(heap, pos):
    endpos = len(heap)
    startpos = pos
    newitem = heap[pos]
    # Bubble up the smaller child until hitting a leaf.
    childpos = 2*pos + 1    # leftmost child position
    while childpos < endpos:
        # Set childpos to index of smaller child.
        rightpos = childpos + 1
        if rightpos < endpos and heap[childpos] >= heap[rightpos]:
            childpos = rightpos
        # Move the smaller child up.
        heap[pos] = heap[childpos]
        pos = childpos
        childpos = 2*pos + 1
    # The leaf at pos is empty now.  Put newitem there, and bubble it up
    # to its final resting place (by sifting its parents down).
    heap[pos] = newitem
    _siftdown(heap, startpos, pos)
