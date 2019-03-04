import logging
import math

from typing import List, Callable, Dict
from simpy import Environment, Event, Resource, Process
from ..messages import *
from ..event_series import EventSeries
from ..agents import Router
from ..constants import DQNROUTE_LOGGER
from .message_env import SimpyMessageEnv

class SimpyRouterEnv(SimpyMessageEnv):
    """
    Conveyor environment which plays the role of outer world and
    handles message passing between routers.
    Passing a message to a router is done via `handle` method.
    """
    pass
