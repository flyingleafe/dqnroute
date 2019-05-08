import random

from typing import List, Tuple
from ..base import *
from ...messages import *

class RandomRouter(Router):
    """
    Simplest router which just sends a packet in a random direction.
    """

    def route(self, sender: AgentId, pkg: Package) -> Tuple[AgentId, List[Message]]:
        return random.choice(self.out_neighbours), []
