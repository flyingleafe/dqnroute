import random

from typing import List, Tuple
from ..base import *
from ...messages import *

class RandomRouter(Router):
    """
    Simplest router which just sends a packet in a random direction.
    """

    def route(self, sender: int, pkg: Package) -> Tuple[int, List[Message]]:
        return random.choice(self.out_neighbours), []

    def handleServiceMsg(self, sender: int, msg: ServiceMessage) -> List[Message]:
        return []
