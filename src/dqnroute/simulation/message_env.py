from typing import List
from simpy import Environment, Event
from ..messages import Message
from ..agents import MessageHandler

class SimpyMessageEnv:
    def __init__(self, env: Environment, handler: MessageHandler):
        self.env = env
        self.handler = handler

    def handle(self, msg: Message) -> Event:
        return self.env.process(self._handleGen(msg))

    def _handleGen(self, msg: Message):
        yield self._msgEvent(msg)
        return [self._msgEvent(m) for m in self.handler.handle(msg)]

    def _msgEvent(self, msg: Message) -> Event:
        raise NotImplementedError()
