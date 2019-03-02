import networkx as nx
import itertools as it

from .messages import *
from .utils import *

class RLAgent:
    """Abstract class representing reinforcement learning unit"""

    def act(self, state, **kwargs):
        pass

    def observe(self, sample):
        pass

class LinkStateHolder:
    def __init__(self):
        self.seq_num = 0
        self.announcements = {}
        self.network_graph = None
        self.removed_links = {}

    def initGraph(self, addr, network, neighbors, link_states):
        self.network_graph = nx.Graph()
        for n in network.keys():
            self.network_graph.add_node(n)
        for (n, data) in neighbors.items():
            alive = link_states[n]['alive']
            if alive:
                self.network_graph.add_edge(addr, n, weight=data['latency'])
            elif self.network_graph.has_edge(addr, n):
                self.network_graph.remove_edge(addr, n)

    def processLSAnnouncement(self, message, nodes):
        from_addr = message.from_addr
        seq = message.seq_num
        data = message.neighbors
        if from_addr not in self.announcements or self.announcements[from_addr][0] < seq:
            self.announcements[from_addr] = (seq, data)
            for (m, params) in data.items():
                m_data = self.announcements.get(m, (0, {}))[1]
                if from_addr in m_data:
                    self.network_graph.add_edge(from_addr, m, **params)
            for m in set(nodes) - set(data.keys()):
                if self.network_graph.has_edge(from_addr, m):
                    self.network_graph.remove_edge(from_addr, m)
            return True
        return False

    def mkLSAnnouncement(self, addr):
        neighbors_data = dict(self.network_graph.adjacency_iter())[addr]
        self.announcements[addr] = (self.seq_num, neighbors_data)
        return LinkStateAnnouncement(self.seq_num, addr, neighbors_data)

    def lsBreakLink(self, u, v):
        self.removed_links[v] = self.network_graph.get_edge_data(u, v)
        self.network_graph.remove_edge(u, v)

    def lsRestoreLink(self, u, v):
        restore_data = self.removed_links[v]
        self.network_graph.add_edge(u, v, **restore_data)
        del self.removed_links[v]

class ConveyorLinkStateHolder:
    def __init__(self):
        self.seq_num = 0
        self.announcements = {}
        self.network_graph = nx.DiGraph()

    def _getSecWeight(self, n, all_sections, speed):
        return all_sections[n]['length'] / speed

    def initGraph(self, all_sections, speed):
        for (sec_num, sec_info) in all_sections.items():
            self.network_graph.add_node(sec_num)
            for n in sec_neighbors_list(sec_info):
                self.network_graph.add_edge(sec_num, n, weight=self._getSecWeight(n, all_sections, speed))

    def processLSAnnouncement(self, message, all_sections, speed):
        something_new = False
        seq = message.seq_num
        for (sec, sec_nbs) in message.section_infos.items():
            if sec not in self.announcements or self.announcements[sec][0] < seq:
                something_new = True
                self.announcements[sec] = (seq, sec_nbs)
                for n in sec_nbs:
                    self.network_graph.add_edge(sec, n, weight=self._getSecWeight(n, all_sections, speed))
                for m in set(all_sections) - set(sec_nbs):
                    if self.network_graph.has_edge(sec, m):
                        self.network_graph.remove_edge(sec, m)
        return something_new

    def mkLSAnnouncement(self, sections):
        section_infos = {}
        adj_dict = dict(self.network_graph.adjacency_iter())
        for sec in sections:
            sec_nbs = list(adj_dict[sec].keys())
            section_infos[sec] = sec_nbs
            self.announcements[sec] = (self.seq_num, sec_nbs)
        return ConveyorLinkStateAnnouncement(self.seq_num, section_infos)

    def lsBreakLink(self, u, v):
        self.network_graph.remove_edge(u, v)

    def lsRestoreLink(self, u, v, all_sections, speed):
        self.network_graph.add_edge(u, v, self._getSecWeight(v, all_sections, speed))
