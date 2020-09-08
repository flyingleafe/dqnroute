import random

from typing import List, Tuple
from ..base import *
from ...messages import *

class RandomRouter(Router):
    """
    Simplest router which just sends a packet in a random direction.
    """

    def route(self, sender: AgentId, pkg: Package, allowed_nbrs: List[AgentId]) -> Tuple[AgentId, List[Message]]:
        return random.choice(allowed_nbrs), []
