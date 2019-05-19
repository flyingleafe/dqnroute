from typing import List, Tuple, Any, Dict
from .utils import *

POS_ROUND_DIGITS = 5

class CollisionException(Exception):
    """
    Thrown when objects on conveyor collide
    """
    def __init__(self, args):
        super().__init__('[{}] Objects {} and {} collided in position {}'
                        .format(args[3], args[0], args[1], args[2]))
        self.obj1 = args[0]
        self.obj2 = args[1]
        self.pos = args[2]
        self.handler = args[3]

class AutomataException(Exception):
    pass

def search_pos(ls: List[Tuple[Any, float]], pos: float,
               offset: float = 0, return_index: bool = False,
               preference: str = 'nearest') -> Tuple[Any, float]:
    assert len(ls) > 0, "what are you gonna find pal?!"
    return binary_search(ls, differs_from(pos, using=lambda p: p[1] + offset),
                         return_index=return_index, preference=preference)

# Automata for simplifying the modelling of objects movement
_model_automata = {
    'pristine': {'resume': 'moving', 'change': 'dirty'},
    'moving': {'pause': 'pristine'},
    'dirty': {'change': 'dirty', 'start_resolving': 'resolving'},
    'resolving': {'change': 'resolving', 'end_resolving': 'resolved'},
    'resolved': {'resume': 'moving'}
}

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
                 checkpoints: List[Tuple[Any, float]],
                 model_id: AgentId = ('world', 0)):
        assert length > 0, "Conveyor length <= 0!"
        assert max_speed > 0, "Conveyor max speed <= 0!"

        checkpoints = sorted(checkpoints, key=lambda p: p[1])
        if len(checkpoints) > 0:
            assert checkpoints[0][1] >= 0, "Checkpoints with position < 0!"
            assert checkpoints[-1][1] < length, "Checkpoint with position >= conveyor length!"
            for i in range(len(checkpoints) - 1):
                assert checkpoints[i][1] < checkpoints[i+1][1], \
                    "Checkpoints with equal positions!"

        # constants
        self.model_id = model_id
        self.checkpoints = checkpoints
        self.checkpoint_positions = {cp: pos for (cp, pos) in checkpoints}
        self.length = length
        self.max_speed = max_speed

        # variables
        self.speed = 0
        self.objects = {}
        self.object_positions = []

        self._state = 'pristine'
        self._resume_time = 0
        self._resolved_events = set()

    def _stateTransfer(self, action):
        try:
            init_state = self._state
            self._state = _model_automata[init_state][action]
        except KeyError:
            print('RAISED')
            raise AutomataException('{}: Invalid action `{}` in state `{}`'.format(self.model_id, action, self._state))

    def checkpointPos(self, cp: Any) -> float:
        if cp[0] == 'conv_end':
            return self.length
        return self.checkpoint_positions[cp]

    def nextCheckpoint(self, pos: float) -> Tuple[Any, float]:
        return search_pos(self.checkpoints, pos, preference='next')

    def nearestObject(self, pos: float, after=None) -> Tuple[Any, float]:
        if after is not None:
            offset = round(after * self.speed, POS_ROUND_DIGITS)
        else:
            offset = 0

        oid, o_pos = search_pos(self.object_positions, pos, offset=offset)
        return self.objects[oid], o_pos

    def working(self) -> bool:
        return self.speed > 0

    def setSpeed(self, speed: float):
        assert speed >= 0 and speed <= self.max_speed, \
            "Speed is not in [0, max_speed]!"
        if self.speed != speed:
            self._stateTransfer('change')
            self.speed = speed

    def putObject(self, obj_id: int, obj: Any, pos: float):
        assert obj_id not in self.objects, "Clashing object ID!"
        pos = round(pos, POS_ROUND_DIGITS)

        if len(self.objects) > 0:
            (n_obj_id, n_pos), n_idx = search_pos(self.object_positions, pos, return_index=True)
            if n_pos == pos:
                raise CollisionException((obj, self.objects[n_obj_id], pos, self.model_id))
            elif n_pos < pos:
                n_idx += 1
        else:
            n_idx = 0

        self.objects[obj_id] = obj
        self.object_positions.insert(n_idx, (obj_id, pos))
        self._stateTransfer('change')

    def removeObject(self, obj_id: int):
        pos_idx = None
        for (i, (oid, pos)) in enumerate(self.object_positions):
            if oid == obj_id:
                pos_idx = i
                break
        if pos_idx is None:
            print('{}: bag#{} not found'.format(self.model_id, obj_id))

        self.object_positions.pop(pos_idx)
        obj = self.objects.pop(obj_id)
        self._stateTransfer('change')
        return obj

    def skipTime(self, time: float, clean_ends=True):
        if time == 0:
            return 0

        d = time * self.speed
        for i in range(len(self.object_positions)):
            obj_id, pos = self.object_positions[i]
            self.object_positions[i] = (obj_id, round(pos + d, POS_ROUND_DIGITS))

        if clean_ends:
            while len(self.object_positions) > 0 and self.object_positions[0][1] < 0:
                obj_id, _ = self.object_positions.pop(0)
                self.objects.pop(obj_id)
            while len(self.object_positions) > 0 and self.object_positions[-1][1] > self.length:
                obj_id, _ = self.object_positions.pop()
                self.objects.pop(obj_id)

        self._stateTransfer('change')
        return d

    def nextEvents(self, obj=None, cps=None, skip_immediate=True,
                   skip_resolved=True) -> List[Tuple[Any, Any, float]]:
        if self.speed == 0:
            return []

        obj = def_list(obj, self.objects.keys())
        obj_positions = [(oid, pos) for (oid, pos) in self.object_positions
                         if oid in obj]

        cps = def_list(cps, self.checkpoint_positions.keys())
        c_points = [(cp, pos) for (cp, pos) in self.checkpoints if cp in cps]
        c_points.append((('conv_end', self.model_id[1]), self.length))

        def _skip_cond(obj_id, cp_idx, pos):
            cp, cp_pos = c_points[cp_idx]
            if skip_resolved and (obj_id, cp) in self._resolved_events:
                return True
            return cp_pos <= pos if skip_immediate else cp_pos < pos

        cp_idx = 0
        events = []
        for (obj_id, pos) in obj_positions:
            assert pos >= 0 and pos <= self.length, \
                "`nextEvents` on conveyor with undefined object positions!"

            while cp_idx < len(c_points) and _skip_cond(obj_id, cp_idx, pos):
                cp_idx += 1

            if cp_idx < len(c_points):
                cp = c_points[cp_idx]
                obj = self.objects[obj_id]
                diff = (cp[1] - pos) / self.speed
                events.append((obj, cp[0], diff))

        events.sort(key=lambda p: p[2])
        return events

    def immediateEvents(self, obj=None, cps=None) -> List[Tuple[Any, Any, float]]:
        return [ev for ev in self.nextEvents(obj=obj, cps=cps, skip_immediate=False) if ev[2] == 0]

    def pickUnresolvedEvent(self) -> Union[Tuple[Any, Any, float], None]:
        assert self.resolving(), "picking event for resolution while not resolving"
        evs = self.nextEvents(skip_immediate=False)
        if len(evs) == 0:
            return None

        obj, cp, diff = evs[0]
        if diff > 0:
            return None

        self._resolved_events.add((obj.id, cp))
        # print('HEEEEEEEY {} {} {}'.format(self.model_id, obj, cp))
        return obj, cp, diff

    def resume(self, time: float):
        self._stateTransfer('resume')
        self._resume_time = time

    def pause(self, time: float):
        self._stateTransfer('pause')
        time_diff = time - self._resume_time
        assert time_diff >= 0, "Pause before resume??"
        self.skipTime(time_diff)

    def startResolving(self):
        self._stateTransfer('start_resolving')

    def endResolving(self):
        self._resolved_events = set()
        self._stateTransfer('end_resolving')

    def pristine(self):
        return self._state == 'pristine'

    def dirty(self):
        return self._state == 'dirty'

    def resolving(self):
        return self._state == 'resolving'

    def resolved(self):
        return self._state == 'resolved'

    def moving(self):
        return self._state == 'moving'


def all_next_events(models: Dict[int, ConveyorModel], **kwargs):
    res = []
    for conv_idx, model in models.items():
        evs = model.nextEvents(**kwargs)
        res = merge_sorted(res, [(conv_idx, ev) for ev in evs],
                           using=lambda p: p[1][2])
    return res

def all_unresolved_events(models: Dict[int, ConveyorModel]):
    while True:
        had_some = False
        for conv_idx, model in models.items():
            if model.resolving():
                ev = model.pickUnresolvedEvent()
                if ev is not None:
                    yield (conv_idx, ev)
                    had_some = True
        if not had_some:
            break
