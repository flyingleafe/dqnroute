import simpy
import random

class SimPyAbstractRouter(object):
    def __init__(self, env, id):
        self.env = env
        self.id = id
        
        self.send_process = env.process(self.sendPkgProcess())
        self.get_process = env.process(self.getPkgProcess())
        self.send_event = env.event()
        self.get_event = env.event()

    def setNeighbours(self, neighbours):
        self.nghbrs = neighbours
        
    def sendPkgProcess(self):
        while True:
            pkg, dest = yield self.send_event
            destRouter = self.nghbrs[dest]
            print("Sending packet: {} -> {} at time {}...".format(self.id, destRouter.id, self.env.now))
            #actual sending simulation
            
            #print("Start sending at: ", self.env.now)
            yield self.env.timeout(random.randint(10, 20))
            yield self.env.process(destRouter.getPackage(self.id, pkg))
            #print("Sent at: ", self.env.now)
            
    # Change broadcast options
    def getPkgProcess(self):
        while True:
            pkg, src = yield self.get_event
            print("[Router {}] Got packet from router {} at time {}".format(self.id, src, self.env.now))
            # proc packet
            #print("Start procing at: ", self.env.now)
            yield self.env.timeout(random.randint(5, 10))
            #print("Proced at: ", self.env.now)

    def route(self, pkg):
        raise NotImplementedError()
    
    def sendPackage(self, pkg):
        dest = self.route(pkg)
        yield self.send_event.succeed(value=(pkg, dest))
        #print("Pkg sent")
        self.send_event = self.env.event()

    def getPackage(self, src, pkg):
        yield self.get_event.succeed(value=(pkg, src))
        self.get_event = self.env.event()

class SimPyDumbRouter(SimPyAbstractRouter):
    def __init__(self, env, id):
        super().__init__(env, id)
        
    def route(self, pkg):
        return random.randint(0, len(self.nghbrs) - 1)
