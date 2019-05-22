from typing import List, Tuple, Any, Dict, Optional
from .utils import *
from .event_series import *

POS_ROUND_DIGITS = 3
SOFT_COLLIDE_SHIFT = 0.2

BAG_RADIUS = 0.5
BAG_MASS = 20
BAG_DENSITY = 40

BELT_WIDTH = 1.5
BELT_UNIT_MASS = 15

SPEED_STEP = 0.1
SPEED_ROUND_DIGITS = 3

MODEL_BELT_LEN = 313.25

_G = 9.8
_FRICTON_COEFF = 0.024
_THETA_1 = 1/(6.48*BELT_WIDTH*BELT_WIDTH*BAG_DENSITY)
_THETA_2_L = _G*_FRICTON_COEFF*BELT_UNIT_MASS
_k_3 = 4000
_THETA_3 = 0.0031
_THETA_4_L = _G*_FRICTON_COEFF
_k_2 = 30

_ETA = 0.8

# THETAS_ORIG = [2.3733e-4, 8566.3, 0.0031, 51.6804]

def _P_Zhang(length, V, T):
    return _THETA_1*V*T*T + (_THETA_2_L*length + _k_3)*V + \
        _THETA_3*T*T/V + (_THETA_4_L*length + _k_2)*T + V*V*T/3.6

def consumption_Zhang(length, speed, n_bags):
    if speed == 0:
        return 0
    Q_g = n_bags * BAG_MASS / length
    T = Q_g * speed * 3.6
    return _P_Zhang(length, speed, T) / _ETA

def consumption_linear(length, speed, n_bags):
    return _G * _FRICTION_COEFF * BAG_MASS * n_bags * speed / _ETA

##
# ^ weird stuff up here ...
#

class EnergySpender(ChangingValue):
    """
    Class which records energy consumption
    """
    def __init__(self, data_series: EventSeries, length, consumption='zhang'):
        super().__init__(data_series)
        if type(consumption) == 'float':
            self._consumption = lambda v, n: consumption
        elif consumption == 'zhang':
            self._consumption = lambda v, n: consumption_Zhang(length, v, n)
        elif consumption == 'linear':
            self._consumption = lambda v, n: consumption_linear(length, v, n)
        else:
            raise Exception('consumption: ' + consumption)

    def update(self, time, speed, n_bags):
        super().update(time, self._consumption(speed, n_bags))


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
               return_index: bool = False,
               preference: str = 'nearest') -> Tuple[Any, float]:
    assert len(ls) > 0, "what are you gonna find pal?!"
    return binary_search(ls, differs_from(pos, using=lambda p: p[1]),
                         return_index=return_index, preference=preference)

def shift(objs, d):
    return [(o, round(p + d, POS_ROUND_DIGITS)) for (o, p) in objs]

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
    def __init__(self, env: DynamicEnv, length: float, max_speed: float,
                 checkpoints: List[Tuple[Any, float]], energy_series: EventSeries = None,
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
        self.env = env
        self.model_id = model_id
        self.checkpoints = checkpoints
        self.checkpoint_positions = {cp: pos for (cp, pos) in checkpoints}
        self.length = length
        self.max_speed = max_speed

        # variables
        self.speed = 0
        self.objects = {}
        self.object_positions = []

        self._spender = None
        if energy_series is not None:
            self._spender = EnergySpender(energy_series, self.length)

        self._state = 'pristine'
        self._resume_time = 0
        self._resolved_events = set()
        self._speed_changes = 0
        self._speed_sum = 0

    def _stateTransfer(self, action):
        try:
            self._state = _model_automata[self._state][action]
            if action == 'change' and self._spender is not None:
                self._spender.update(self.env.time(), self.speed, len(self.objects))
        except KeyError:
            print('RAISED')
            raise AutomataException(
                '{}: Invalid action `{}` in state `{}`;\n  speed - {}m/s\n  cps - {};\n  objs - {}'
                .format(self.model_id, action, self._state, self.speed,
                        self.checkpoints, self.object_positions))

    def checkpointPos(self, cp: Any) -> Optional[float]:
        if cp[0] == 'conv_end':
            return self.length
        return self.checkpoint_positions.get(cp, None)

    def nextCheckpoint(self, pos: float) -> Tuple[Any, float]:
        return search_pos(self.checkpoints, pos, preference='next')

    def nearestObject(self, pos: float, after=None, speed=None, not_exact=False,
                      preference='nearest') -> Optional[Tuple[Any, float]]:
        if len(self.object_positions) == 0:
            return None

        if after is not None:
            if speed is None:
                speed = self.speed
            objs = shift(self.object_positions, after * speed)
        else:
            objs = self.object_positions

        res = search_pos(objs, pos, preference=preference, return_index=True)
        if res is not None:
            (oid, o_pos), idx = res
            if not_exact and o_pos == pos:
                if preference == 'prev':
                    idx -= 1
                elif preference == 'next':
                    idx += 1
                else:
                    raise Exception('please dont use nearest with not exact')
                if idx < 0 or idx > len(objs):
                    return None
                oid, o_pos = objs[idx]
            return self.objects[oid], o_pos
        return None

    def working(self) -> bool:
        return self.speed > 0

    def setSpeed(self, speed: float):
        assert speed >= 0 and speed <= self.max_speed, \
            "Speed is not in [0, max_speed]!"
        if self.speed != speed:
            self._stateTransfer('change')
            self.speed = speed
            self._speed_changes += 1
            self._speed_sum += speed

    def putObject(self, obj_id: int, obj: Any, pos: float, soft_collide=True, return_nearest=False):
        assert obj_id not in self.objects, "Clashing object ID!"
        pos = round(pos, POS_ROUND_DIGITS)

        nearest = None
        if len(self.objects) > 0:
            (n_obj_id, n_pos), n_idx = search_pos(self.object_positions, pos, return_index=True)
            if n_pos == pos:
                if soft_collide:
                    print('{}: TRUE COLLISION: #{} and #{} on {}'.format(self.model_id, obj_id, n_obj_id, pos))
                    i = n_idx
                    p_pos = pos
                    while i < len(self.object_positions) and self.object_positions[i][1] >= p_pos:
                        p_pos = round(p_pos + SOFT_COLLIDE_SHIFT, POS_ROUND_DIGITS)
                        self.object_positions[i] = (self.object_positions[i][0], p_pos)
                        i += 1
                else:
                    raise CollisionException((obj, self.objects[n_obj_id], pos, self.model_id))
            elif n_pos < pos:
                n_idx += 1
            nearest = (n_obj_id, n_pos)
        else:
            n_idx = 0

        self.objects[obj_id] = obj
        self.object_positions.insert(n_idx, (obj_id, pos))
        self._stateTransfer('change')

        if return_nearest:
            return nearest

    def objPos(self, obj_id: int):
        for (oid, pos) in self.object_positions:
            if oid == obj_id:
                return pos
        return None

    def removeObject(self, obj_id: int):
        pos_idx = None
        for (i, (oid, pos)) in enumerate(self.object_positions):
            if oid == obj_id:
                pos_idx = i
                break

        self.object_positions.pop(pos_idx)
        obj = self.objects.pop(obj_id)
        self._stateTransfer('change')
        return obj

    def shift(self, d):
        self.object_positions = shift(self.object_positions, d)

    def skipTime(self, time: float, clean_ends=True):
        if time == 0:
            return 0

        self._stateTransfer('change')
        d = time * self.speed
        if len(self.objects) == 0:
            return d

        last_pos_orig = self.object_positions[-1][1]
        self.shift(d)

        if clean_ends:
            while len(self.object_positions) > 0 and self.object_positions[0][1] < 0:
                obj_id, _ = self.object_positions.pop(0)
                self.objects.pop(obj_id)
            while len(self.object_positions) > 0 and self.object_positions[-1][1] > self.length:
                obj_id, pos = self.object_positions.pop()
                # print('HAHA NE ZHDAL? A VOT, NAHUI IDET: {}, {} -> {}; {}s, {}m/s'.format(obj_id, last_pos_orig, pos, time, self.speed))
                self.objects.pop(obj_id)

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

    def resume(self):
        self._stateTransfer('resume')
        self._resume_time = self.env.time()

    def pause(self):
        self._stateTransfer('pause')
        time_diff = self.env.time() - self._resume_time
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
            if model.dirty():
                model.startResolving()
            if model.resolving():
                ev = model.pickUnresolvedEvent()
                if ev is not None:
                    yield (conv_idx, ev)
                    had_some = True
        if not had_some:
            break

def find_max_speed(up_model, up_pos, up_speed, dist_to_end, max_speed, bag_radius=BAG_RADIUS):
    speed = max_speed
    while speed > 0:
        end_time = dist_to_end / speed
        up_nearest = up_model.nearestObject(up_pos, after=end_time, speed=up_speed)
        if up_nearest is None:
            break

        _, obj_pos = up_nearest
        if abs(up_pos - obj_pos) >= 2*bag_radius:
            break
        elif dist_to_end == 0:
            return 0
        speed = round(speed - SPEED_STEP, SPEED_ROUND_DIGITS)

    return max(0, speed)
