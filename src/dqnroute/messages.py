from functools import total_ordering
# import pandas as pd
# import numpy as np

class Message:
    """
    Base class for all messages used in the system
    """

    def __init__(self, contents):
        self.contents = contents

    def getContents(self):
        return self.contents

class UnsupportedMessageType(Exception):
    """
    Exception which is thrown by message handlers on encounter of
    unknown message type
    """
    pass

class InitMessage(Message):
    """
    Message which router receives as environment starts
    """
    pass

class InMessage(Message):
    """
    Wrapped message which has came from the outside.
    """
    def __init__(self, sender: int, contents: Message):
        super().__init__(contents)
        self.sender = sender

class OutMessage(Message):
    """
    Wrapped message which is sent to a neighbor through the interface
    with given ID.
    """
    def __init__(self, recipient: int, contents: Message):
        super().__init__(contents)
        self.recipient = recipient

def repackMsg(sender: int, msg: OutMessage) -> InMessage:
    """
    Creates an incoming message for recipient from outgoing message
    for sender
    """
    return InMessage(sender, msg.getContents())

class ServiceMessage(Message):
    """
    Message which does not contain a package and, hence,
    contains no destination.
    """
    pass

# Packages
@total_ordering
class Package:
    def __init__(self, pkg_id, size, dst, start_time, state_size, contents):
        self.id = pkg_id
        self.size = size
        self.dst = dst
        self.start_time = start_time
        self.route = None
        self.contents = contents
        # self.rnn_state = (np.zeros((1, state_size)),
        #                   np.zeros((1, state_size)))

    # def route_add(self, data, cols):
    #     if self.route is None:
    #         self.route = pd.DataFrame(columns=cols)
    #     self.route.loc[len(self.route)] = data

    def __hash__(self):
        return hash((self.id, self.contents))

    def __eq__(self, other):
        return self.id == other.id

    def __lt__(self, other):
        return self.id < other.id

# class Bag(Package):
#     def __init__(self, bag_id, dst, start_time, prev_time, state_size, contents):
#         super().__init__(bag_id, 0, dst, start_time, state_size, contents)
#         self.prev_time = prev_time
#         self.energy_spent = 0

class PkgMessage(Message):
    """
    Message which contains a package which has a particular
    destination.
    """
    def __init__(self, pkg: Package):
        super().__init__(pkg)

class PkgReceivedMessage(Message):
    """
    Message which router uses in case it was the package's
    destination
    """
    def __init__(self, pkg: Package):
        super().__init__(pkg)

class AddLinkMessage(Message):
    """
    Message which router receives when a link is connected (or restored).
    """
    def __init__(self, to: int, params={}):
        super().__init__({"to": to, "params": params})

class RemoveLinkMessage(Message):
    """
    Message which router receives when link breakage is detected
    """
    def __init__(self, to: int):
        super().__init__(to)

#
# Service messages
#

class RewardMsg(ServiceMessage):
    def __init__(self, action_id: int, reward_data):
        super().__init__({
            "action_id": action_id,
            "reward_data": reward_data,
        })

class NetworkRewardMsg(RewardMsg):
    def __init__(self, pkg_id: int, time_received: float, Q: float):
        super().__init__(pkg_id, (time_received, Q))

class ConveyorRewardMsg(RewardMsg):
    def __init__(self, bag_id: int, time_processed: float, energy_gap: float, Q: float):
        super().__init__(bag_id, (time_received, energy_gap, Q))

class LSAnnouncementMsg(ServiceMessage):
    def __init__(self, node: int, seq: int, neighbours):
        super().__init__({
            "node": node,
            "seq": seq,
            "neighbours": neighbours
        })
