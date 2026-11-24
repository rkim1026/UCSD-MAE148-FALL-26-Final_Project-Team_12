"""Scenario simulator that imports the real ring_gap + state machine.

One human-driven car on the ring, RoboCar approaches / yields / merges /
follows. LiDAR is modelled as noisy angle+speed observations at 10 Hz so
the policy sees the same numbers as on the car.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import math
import random

from roundabout_pkg.ring_gap import (
    GapParams,
    RingObservation,
    TrackGeometry,
    gap_time,
    remaining_angle_to_conflict,
    zone_busy_from_track,
)
from roundabout_pkg.state_machine import RightOfWayLogic, RowState


@dataclass
class HumanCar:
    angle: float = math.pi  # start opposite the conflict point
    speed: float = 0.8  # m/s
    behaviour: str = "cruise"  # stop | slow | cruise | fast | random

    def target_speed(self) -> float:
        table = {"stop": 0.0, "slow": 0.4, "cruise": 0.8, "fast": 1.2}
        if self.behaviour == "random":
            return random.choice([0.0, 0.4, 0.8, 1.2])
        return table.get(self.behaviour, 0.8)


@dataclass
class EgoCar:
    # Position along entry then on ring. Phase: approach | on_ring | exited
    phase: str = "approach"
    # Distance to yield line along the entry road (m). 0 = at line.
    dist_to_line: float = 1.5
    angle_on_ring: float = 0.0  # set to conflict_angle on merge
    speed: float = 0.0
    length: float = 0.35  # m, for clearing the conflict zone


@dataclass
class SimStats:
    entries: int = 0
    collisions: int = 0
    human_braked: int = 0
    wait_times: list = field(default_factory=list)
    _waiting: bool = False
    _wait_start: float = 0.0
    _entered_at: Optional[float] = None
    _human_speed_at_entry: Optional[float] = None


@dataclass
class ScenarioSim:
    track: TrackGeometry = field(default_factory=TrackGeometry)
    params: GapParams = field(default_factory=GapParams)
    dt: float = 0.1  # 10 Hz
    human: HumanCar = field(default_factory=HumanCar)
    ego: EgoCar = field(default_factory=EgoCar)
    logic: RightOfWayLogic = field(default_factory=RightOfWayLogic)
    stats: SimStats = field(default_factory=SimStats)
    # Simulated LiDAR noise on measured speed (±0.2 m/s typical).
    speed_noise_std: float = 0.2
    angle_noise_std: float = math.radians(2.0)
    drop_scan_prob: float = 0.02
    time: float = 0.0
    exit_lap_fraction: float = 0.25  # exit after this fraction of a lap
    _human_brake_latched: bool = False

    def __post_init__(self) -> None:
        self.logic.params = self.params
        self.logic.use_continuous_following = True
        self.logic.reset()

    def reset(
        self,
        *,
        human_behaviour: str = "cruise",
        human_angle: Optional[float] = None,
        ego_dist: float = 1.5,
    ) -> Dict:
        self.time = 0.0
        self.human = HumanCar(
            angle=math.pi if human_angle is None else human_angle,
            behaviour=human_behaviour,
            speed=0.8,
        )
        self.ego = EgoCar(dist_to_line=ego_dist, speed=0.0, phase="approach")
        self.logic.reset()
        self.stats = SimStats()
        self._human_brake_latched = False
        self._collision_latched = False
        return self._obs_dict(self._sense())

    def _true_lead_distance(self) -> float:
        if self.ego.phase != "on_ring":
            return float("inf")
        dtheta = remaining_angle_to_conflict(self.ego.angle_on_ring, self.human.angle)
        # Distance along lane from ego bumper to human bumper (approx).
        # Using remaining angle from ego to human CCW:
        from roundabout_pkg.ring_gap import wrap_angle

        delta = wrap_angle(self.human.angle - self.ego.angle_on_ring)
        if delta < 0.0:
            delta += 2.0 * math.pi
        return max(0.0, delta * self.track.lane_center_radius - self.ego.length)

    def _sense(self) -> RingObservation:
        if random.random() < self.drop_scan_prob:
            # Dropped scan → keep previous conservative: busy unknown → treat empty
            return RingObservation(
                at_yield_line=self.ego.dist_to_line <= 0.05 and self.ego.phase == "approach",
                front_clear=True,
            )

        noisy_angle = self.human.angle + random.gauss(0.0, self.angle_noise_std)
        noisy_speed = max(0.0, self.human.speed + random.gauss(0.0, self.speed_noise_std))
        phi = remaining_angle_to_conflict(noisy_angle, self.track.conflict_angle)
        gt = gap_time(phi, noisy_speed, self.track.lane_center_radius)

        # Conflict occupied if human near conflict angle.
        from roundabout_pkg.ring_gap import wrap_angle

        occupied = abs(wrap_angle(self.human.angle - self.track.conflict_angle)) < math.radians(
            20.0
        )
        busy = zone_busy_from_track(phi, noisy_speed, self.track, conflict_occupied=occupied)

        at_line = self.ego.phase == "approach" and self.ego.dist_to_line <= 0.05
        past = self.ego.phase == "on_ring" and (
            remaining_angle_to_conflict(self.track.conflict_angle, self.ego.angle_on_ring)
            > math.radians(25.0)
            or self.ego.angle_on_ring != self.track.conflict_angle
        )
        # After merge, mark past once we have travelled a short arc.
        if self.ego.phase == "on_ring":
            travelled = wrap_angle(self.ego.angle_on_ring - self.track.conflict_angle)
            if travelled < 0.0:
                travelled += 2.0 * math.pi
            past = travelled > math.radians(15.0)

        exit_mark = False
        if self.ego.phase == "on_ring":
            travelled = wrap_angle(self.ego.angle_on_ring - self.track.conflict_angle)
            if travelled < 0.0:
                travelled += 2.0 * math.pi
            exit_mark = travelled >= self.exit_lap_fraction * 2.0 * math.pi

        lead = self._true_lead_distance()
        # Collision check: gap under 0.1 m while both on ring.
        front_clear = not (
            self.ego.phase == "on_ring" and lead < 0.10
        )

        return RingObservation(
            at_yield_line=at_line,
            zone_busy=busy,
            gap_time=gt,
            lead_distance=lead,
            phi=phi,
            car_speed=noisy_speed,
            own_speed=self.ego.speed,
            exit_mark_seen=exit_mark,
            past_conflict=past,
            front_clear=front_clear,
        )

    def step(
        self,
        *,
        human_behaviour: Optional[str] = None,
        human_speed_override: Optional[float] = None,
    ) -> Tuple[Dict, float, bool, Dict]:
        if human_behaviour is not None:
            self.human.behaviour = human_behaviour

        prev_state = self.logic.state
        obs = self._sense()
        cmd = self.logic.step(obs)

        # Track wait time and count every merge (direct or after yielding).
        if cmd.state == RowState.YIELD:
            if not self.stats._waiting:
                self.stats._waiting = True
                self.stats._wait_start = self.time
        if cmd.state == RowState.MERGE and prev_state != RowState.MERGE:
            if self.stats._waiting:
                self.stats.wait_times.append(self.time - self.stats._wait_start)
                self.stats._waiting = False
            self.stats.entries += 1
            self.stats._entered_at = self.time
            self.stats._human_speed_at_entry = self.human.speed
            self._human_brake_latched = False

        # Integrate ego.
        self.ego.speed = cmd.speed
        if self.ego.phase == "approach":
            self.ego.dist_to_line = max(0.0, self.ego.dist_to_line - cmd.speed * self.dt)
            if cmd.state == RowState.MERGE and self.ego.dist_to_line <= 0.05:
                self.ego.phase = "on_ring"
                self.ego.angle_on_ring = self.track.conflict_angle
        elif self.ego.phase == "on_ring":
            dtheta = (cmd.speed * self.dt) / self.track.lane_center_radius
            self.ego.angle_on_ring = (self.ego.angle_on_ring + dtheta) % (2.0 * math.pi)
            if cmd.state == RowState.EXIT or cmd.state == RowState.DONE:
                self.ego.phase = "exited"
                self.logic.finish_exit()
            # Auto-finish exit after mark.
            if obs.exit_mark_seen and cmd.state == RowState.EXIT:
                self.ego.phase = "exited"
                self.logic.finish_exit()
        elif self.ego.phase == "exited":
            self.ego.speed = 0.0

        # Integrate human (simple accel toward target, backs off if ego cuts in).
        target = (
            human_speed_override
            if human_speed_override is not None
            else self.human.target_speed()
        )
        # If ego just entered and is close ahead of human → human brakes.
        if (
            self.ego.phase == "on_ring"
            and self.stats._entered_at is not None
            and (self.time - self.stats._entered_at) < 3.0
        ):
            lead_for_human = self._lead_human_to_ego()
            if lead_for_human < 0.6:
                target = min(target, 0.2)
                if not self._human_brake_latched and self.stats._human_speed_at_entry and self.stats._human_speed_at_entry > 0.3:
                    self.stats.human_braked += 1
                    self._human_brake_latched = True

        accel = 1.5  # m/s^2
        if self.human.speed < target:
            self.human.speed = min(target, self.human.speed + accel * self.dt)
        else:
            self.human.speed = max(target, self.human.speed - accel * self.dt)

        dtheta_h = (self.human.speed * self.dt) / self.track.lane_center_radius
        self.human.angle = (self.human.angle + dtheta_h) % (2.0 * math.pi)

        # Collision bookkeeping (one count per contact episode).
        if self.ego.phase == "on_ring" and self._true_lead_distance() < 0.10:
            if not getattr(self, "_collision_latched", False):
                self.stats.collisions += 1
                self._collision_latched = True
        else:
            self._collision_latched = False

        self.time += self.dt
        done = self.ego.phase == "exited" or self.logic.state == RowState.DONE
        reward = 0.0
        if done and self.stats.collisions == 0:
            reward = 1.0
        elif self.stats.collisions:
            reward = -1.0

        info = {
            "state": cmd.state.name,
            "gap_time": obs.gap_time,
            "phi": obs.phi,
            "lead": obs.lead_distance,
            "entries": self.stats.entries,
            "collisions": self.stats.collisions,
            "human_braked": self.stats.human_braked,
            "avg_wait": (
                sum(self.stats.wait_times) / len(self.stats.wait_times)
                if self.stats.wait_times
                else None
            ),
        }
        return self._obs_dict(obs), reward, done, info

    def _lead_human_to_ego(self) -> float:
        from roundabout_pkg.ring_gap import wrap_angle

        if self.ego.phase != "on_ring":
            return float("inf")
        delta = wrap_angle(self.ego.angle_on_ring - self.human.angle)
        if delta < 0.0:
            delta += 2.0 * math.pi
        return delta * self.track.lane_center_radius

    @staticmethod
    def _obs_dict(obs: RingObservation) -> Dict:
        return {
            "at_yield_line": obs.at_yield_line,
            "zone_busy": obs.zone_busy,
            "gap_time": obs.gap_time,
            "lead_distance": obs.lead_distance,
            "phi": obs.phi,
            "car_speed": obs.car_speed,
            "own_speed": obs.own_speed,
            "exit_mark_seen": obs.exit_mark_seen,
            "past_conflict": obs.past_conflict,
            "front_clear": obs.front_clear,
        }


def run_batch(
    n: int = 20,
    behaviours: Optional[list] = None,
    seed: int = 0,
) -> Dict:
    """Batch-test the real decision code; returns aggregate stats."""
    random.seed(seed)
    behaviours = behaviours or ["slow", "cruise", "fast", "stop"]
    total = SimStats()
    for i in range(n):
        beh = behaviours[i % len(behaviours)]
        # Stagger human angle so arrival timing varies.
        sim = ScenarioSim()
        sim.reset(human_behaviour=beh, human_angle=(i * 0.7) % (2.0 * math.pi))
        for _ in range(400):
            _, _, done, _ = sim.step()
            if done:
                break
        total.entries += sim.stats.entries
        total.collisions += sim.stats.collisions
        total.human_braked += sim.stats.human_braked
        total.wait_times.extend(sim.stats.wait_times)
    return {
        "runs": n,
        "entries": total.entries,
        "collisions": total.collisions,
        "human_braked": total.human_braked,
        "avg_wait": (
            sum(total.wait_times) / len(total.wait_times) if total.wait_times else None
        ),
    }


def main() -> None:
    stats = run_batch(30)
    print("Scenario batch results:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
