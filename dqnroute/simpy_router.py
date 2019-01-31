import simpy
import random

class SimPyAbstractRouter(object):
	def __init__(self, env, id, neighbours):
		self.env = env
		self.id = id
		self.nghbrs = neighbours
		self.get_process = env.process(self.getPkg())
		
	# Change broadcast options
	def getPkg(self):
		while True:
			print("[Router %d] Waiting for the packet" % self.id)
			pass
	
	def __route(self, pkg):
		pass
	
	# ~...Store is better
	def sendRandomPackage(self, pkg):
		pass
		
class SimPyDumbRouter(SimPyAbstractRouter):
	def __init__(self, env, neighbours):
		super().__init__(env, neighbours)
		
	def __route(self, pkg):
		return self.nghbrs[random.randint(0, len(nghbrs) - 1)]