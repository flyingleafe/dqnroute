import unittest

from event_queue import *
from collections import namedtuple

Timed = namedtuple('Timed', ['time'])

class TestEventQueue(unittest.TestCase):
    def setUp(self):
        self.queue = EventQueue()
        self.queue.push(Timed(123))
        self.queue.push(Timed(342))
        self.queue.push(Timed(31))
        self.queue.push(Timed(4124))
        self.queue.push(Timed(4141))
        self.queue.push(Timed(13))
        self.queue.push(Timed(41))

    def test_peek(self):
        self.assertEqual(self.queue.peek().time, 13)
        self.queue.pop()
        self.assertEqual(self.queue.peek().time, 31)

    def test_pop(self):
        self.assertEqual(self.queue.pop().time, 13)
        self.assertEqual(self.queue.pop().time, 31)
        self.queue.pop()
        self.assertEqual(self.queue.pop().time, 123)

    def test_earlier_than(self):
        res = self.queue.earlier_than(Timed(123))
        self.assertEqual(set(res), set(map(Timed, [13, 31, 41, 123])))

if __name__ == '__main__':
    unittest.main()
