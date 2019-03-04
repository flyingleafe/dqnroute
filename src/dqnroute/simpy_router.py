import simpy
import random
import logging

class SimPyAbstractRouter(object):
    def __init__(self, env, id):
        self.env = env
        self.id = id
        self.logger = logging.getLogger("SimPyAbstractRouter")

    def setNeighbours(self, neighbours):
        self.neighbours = neighbours

    def __sendSimulation(self, pkg, dest):
        yield self.env.timeout(random.randint(5, 10))
        # manual trigger getter on destination router
        dest.getPackage(self.id, pkg)

    def __sendPkg(self, pkg, dest):
        print("Sending packet: {} -> {} at time {}...".format(self.id, dest.id, self.env.now))

        if self.id == dest.id:
            self.logger.info("Got source router {} as the destination router".format(self.id))
            # actual process of pkg & empty yielding
            yield self.env.timeout(0)
        else:
            # actual sending process & simulation
            self.env.process(self.__sendSimulation(pkg, dest))
        self.logger.debug("Pkg sent from router {} at time {}".format(self.id, self.env.now))

    def __getPkg(self, pkg, src):
        print("Got packet: {} <- {} at time {}".format(self.id, src, self.env.now))

        # process packet & empty yielding
        yield self.env.timeout(0)
        if pkg.dst != self.id:
            self.logger.info("Package destination is not reached on router {}, going to the next router...".format(self.id))
            self.sendPackage(pkg)
        else:
            self.logger.info("Package destination is reached on router {}".format(self.id))
        self.logger.debug("Proced on router {} at time {}".format(self.id, self.env.now))

    def route(self, pkg):
        raise NotImplementedError()

    def sendPackage(self, pkg):
        if self.id == pkg.dst:
            self.logger.info("Got source router {} as the destination router".format(self.id))
            # make processing of this situation
        else:
            dest = self.route(pkg)
            self.env.process(self.__sendPkg(pkg, dest))

    def getPackage(self, src, pkg):
        self.env.process(self.__getPkg(pkg, src))

class SimPyDumbRouter(SimPyAbstractRouter):
    def __init__(self, env, id):
        super().__init__(env, id)

    def route(self, pkg):
        return self.neighbours[random.randint(0, len(self.neighbours) - 1)]
