"""
Module which defines a class which is able to tell current time.
"""
import time

from simpy import Environment

class TimeEnv(object):
    def currentTime(self) -> int:
        raise NotImplementedError()

class SimpyTimeEnv(TimeEnv):
    def __init__(self, env: Environment):
        self.env = env

    def currentTime(self) -> int:
        return self.env.now

class RealTimeEnv(TimeEnv):
    def __init__(self, precision: str = "ms"):
        if precision == "s":
            self._mult = 1
        elif precision == "ms":
            self._mult = 1000
        elif precision == "mcs":
            self._mult = 1000000
        else:
            raise Exception("RealTimeEnv: unknown precision " + precision)

    def currentTime(self) -> int:
        return int(time.time() * self._mult)
