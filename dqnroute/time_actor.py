import copy
from thespian.actors import *
from messages import *
from event_queue import EventQueue

class AbstractTimeActor(Actor):
    """Actor which receives sycnchronization ticks and send timed messages accordingly"""

    def receiveMessage(self, message, sender):
        if isinstance(message, InitMsg):
            self.initialize(message, sender)
        elif isinstance(message, EventMsg):
            self.handleIncomingEvent(message, sender)
        elif isinstance(message, ServiceMsg):
            self.receiveServiceMsg(message, sender)
        elif isinstance(message, TickMsg):
            self.handleTick(message.time)
        else:
            pass

    def sendServiceMsg(self, targetAddress, message):
        self.send(targetAddress, message)

    def sendEvent(self, targetAddress, event):
        """In AbstractTimeActor event is just sent to target. Subject to redefinition"""
        self.send(targetAddress, event)

    def resendEventDelayed(self, targetAddress, event, delay):
        e = copy.copy(event)
        e.sender = self.myAddress
        e.time += delay
        self.sendEvent(targetAddress, e)

    def handleIncomingEvent(self, message, sender):
        pass

    def handleTick(self, time):
        pass

    def initialize(self, message, sender):
        pass

    def receiveServiceMsg(self, message, sender):
        pass

class TimeActor(AbstractTimeActor):
    """
    Particular implementation of `AbstractTimeActor`, which maintains
    priority queues for incoming and outgoing events
    """

    def __init__(self):
        self.current_time = 0
        self.event_queue = EventQueue()

    def handleIncomingEvent(self, message, sender):
        message.sender = sender
        self.event_queue.push(message)

    def handleTick(self, time):
        """
        Handling events in the sequence they should be evaluated
        """
        try:
            while self.event_queue.peek().time <= time:
                e = self.event_queue.pop()
                self.current_time = e.time
                self.processEvent(e)
        except IndexError:
            pass
        self.current_time = time

    def processEvent(self, event):
        pass

class Synchronizer(Actor):
    """Actor which sends `TickMsg`s"""

    def __init__(self):
        self.period = None
        self.delta = None
        self.current_time = 0
        self.targets = []

    def receiveMessage(self, message, sender):
        if isinstance(message, SynchronizerInitMsg):
            self.period = message.period
            self.delta = message.delta
            self.targets = message.targets
            self._delayTick()
        elif isinstance(message, WakeupMessage):
            self._sendTicks()
            self._delayTick()
        else:
            pass

    def _delayTick(self):
        self.wakeupAfter(self.period)

    def _sendTicks(self):
        tick = TickMsg(self.current_time)
        for t in self.targets:
            self.send(t, tick)
        self.current_time += self.delta
