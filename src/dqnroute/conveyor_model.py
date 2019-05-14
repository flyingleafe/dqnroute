from typing import List, Tuple, Any
from .utils import *

class CollisionException(Exception):
    """
    Thrown when objects on conveyor collide
    """
    def __init__(self, obj1, obj2, pos):
        super().__init__('Objects {} and {} collided in position {}'
                        .format(obj1, obj2, pos))
        self.obj1 = obj1
        self.obj2 = obj2
        self.pos = pos


class ConveyorModel:
    """
    Datatype which allows for modeling the conveyor with
    objects moving on top of it.

    A conveyor is modeled as a line with some checkpoints
    placed along it. Only one checkpoint can occupy a given position.

    Objects can be placed on a conveyor. All the objects on a conveyor
    move with the same speed - the current speed of a conveyor. Current speed
    of a conveyor can be changed.

    We want to have the following operations:
    - put an object to a conveyor (checking for collisions)
    - remove an object from a conveyor
    - change the conveyor speed
    - update the positions of objects as if time T has passed with a current
      conveyor speed (T might be negative, moving objects backwards)
    - calculate time T until earliest event of object reaching a checkpoint
      (or the end of the conveyor), and also return those checkpoint and object
    """
    def __init__(self, length: float, max_speed: float,
                 checkpoints: List[Tuple[float, Any]]):
        assert length > 0, "Conveyor length <= 0!"
        assert max_speed > 0, "Conveyor max speed <= 0!"

        checkpoints = sorted(checkpoints, key=lambda p: p[0])
        assert checkpoints[0][0] > 0, "Checkpoints with position <= 0!"
        assert checkpoints[-1][0] < length, "Checkpoint with position >= conveyor length!"
        for i in range(len(checkpoints) - 1):
            assert checkpoints[i][0] < checkpoints[i+1][0], \
                "Checkpoints with equal positions!"

        # constants
        self.checkpoints = checkpoints
        self.length = length
        self.max_speed = max_speed

        # variables
        self.speed = 0
        self.objects = {}
        self.object_positions = []

    def setSpeed(self, speed: float):
        assert speed >= 0 and speed <= self.max_speed, \
            "Speed is not in [0, max_speed]!"
        self.speed = speed

    def putObject(self, obj_id: int, obj: Any, pos: float):
        assert obj_id not in self.objects, "Clashing object ID!"

        n_idx, (n_pos, n_obj_id) = self._findNearestObj(obj, pos, return_index=True)
        if n_pos == pos:
            raise CollisionException(obj, self.objects[n_obj_id], pos)
        elif n_pos < pos:
            n_idx += 1

        self.objects[obj_id] = obj
        self.object_positions.insert(n_idx, (pos, obj_id))

    def removeObject(self, obj_id: int):
        pos_idx = None
        for (i, (pos, oid)) in enumerate(self.object_positions):
            if oid == obj_id:
                pos_idx = i
                break
        self.object_positions.pop(pos_idx)
        return self.objects.pop(obj_id)

    def skipTime(self, time: float, clean_ends=False):
        d = time * self.speed
        for i in range(len(self.object_positions)):
            self.object_positions[i][0] += d

        if clean_ends:
            while self.object_positions[0][0] < 0:
                _, obj_id = self.object_positions.pop(0)
                self.objects.pop(obj_id)
            while self.object_positions[-1][0] > self.length:
                _, obj_id = self.object_positions.pop()
                self.objects.pop(obj_id)

    def nextEvent(self, skip_immediate=True) -> Tuple[Any, Any, float]:
        cp_idx = 0
        events = []
        for (pos, obj_id) in self.object_positions:
            while self.checkpoints[cp_idx][0] < pos:
                cp_idx += 1
            if skip_immediate and pos == self.checkpoints[cp_idx][0]:
                cp_idx += 1
            diff = self.checkpoints[cp_idx][0] - pos
            events.append((self.objects[obj_id], self.checkpoints[cp_idx][1], diff))
        events.sort(key=lambda p: p[2])
        return events[0]

    def _findNearestObj(self, pos, return_index=False) -> Tuple[float, int]:
        return binary_search(self.object_positions,
                             differs_from(pos, using=lambda p: p[0]),
                             return_index=return_index)
