"""Five-state right-of-way machine + optional learned veto.

States: APPROACH → YIELD → MERGE → FOLLOW → EXIT  (+ STOP / DONE)

Transitions match the numbered arrows in the Team 12 build guide.
Learned models can only veto a "go"; they never force an entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional

from roundabout_pkg.ring_gap import (
    GapParams,
    RingObservation,
    following_speed,
    gap_rule_go,
)


class RowState(Enum):
    APPROACH = auto()
    YIELD = auto()
    MERGE = auto()
    FOLLOW = auto()
    EXIT = auto()
    STOP = auto()
    DONE = auto()


# Optional veto: returns True if going is allowed.
GoVeto = Callable[[RingObservation], bool]


@dataclass
class RowCommand:
    state: RowState
    speed: float  # m/s desired (steering comes from the lane follower)
    go_votes: int = 0
    rule_go: bool = False
    model_go: bool = True


@dataclass
class RightOfWayLogic:
    params: GapParams = field(default_factory=GapParams)
    state: RowState = RowState.APPROACH
    go_votes: int = 0
    veto: Optional[GoVeto] = None
    merge_speed: float = 0.5  # m/s while entering
    approach_speed: Optional[float] = None  # defaults to cruise
    use_continuous_following: bool = False

    def reset(self) -> None:
        self.state = RowState.APPROACH
        self.go_votes = 0

    def _combined_go(self, obs: RingObservation) -> tuple[bool, bool, bool]:
        rule = gap_rule_go(obs, self.params)
        model = True if self.veto is None else bool(self.veto(obs))
        return rule and model, rule, model

    def step(self, obs: RingObservation) -> RowCommand:
        # Global abort: E-stop / close obstacle.
        if not obs.front_clear:
            self.state = RowState.STOP
            self.go_votes = 0
            return RowCommand(self.state, 0.0, self.go_votes, False, True)

        cruise = (
            self.params.cruise_speed
            if self.approach_speed is None
            else self.approach_speed
        )

        if self.state == RowState.DONE:
            return RowCommand(self.state, 0.0, self.go_votes)

        if self.state == RowState.STOP:
            # Hold until the operator / clear path restarts via reset().
            return RowCommand(self.state, 0.0, self.go_votes)

        if self.state == RowState.APPROACH:
            combined, rule, model = self._combined_go(obs)
            if obs.at_yield_line:
                if combined:
                    # (1) Gap already good → merge without stopping.
                    self.state = RowState.MERGE
                    self.go_votes = 0
                    return RowCommand(self.state, self.merge_speed, 0, rule, model)
                # (2) Gap too small / zone busy → yield.
                self.state = RowState.YIELD
                self.go_votes = 0
                return RowCommand(self.state, 0.0, 0, rule, model)
            return RowCommand(self.state, cruise, self.go_votes, False, True)

        if self.state == RowState.YIELD:
            combined, rule, model = self._combined_go(obs)
            if combined:
                self.go_votes += 1
            else:
                self.go_votes = 0
            # (3) Need N consecutive go votes before merging.
            if self.go_votes >= self.params.go_scans_required:
                self.state = RowState.MERGE
                self.go_votes = 0
                return RowCommand(self.state, self.merge_speed, 0, rule, model)
            return RowCommand(self.state, 0.0, self.go_votes, rule, model)

        if self.state == RowState.MERGE:
            combined, rule, model = self._combined_go(obs)
            # (4) Abort back to yield if a car appears before we clear the line.
            if obs.at_yield_line and not combined:
                self.state = RowState.YIELD
                self.go_votes = 0
                return RowCommand(self.state, 0.0, 0, rule, model)
            # (5) Past the conflict zone → follow on the ring.
            if obs.past_conflict:
                self.state = RowState.FOLLOW
                return RowCommand(
                    self.state,
                    following_speed(
                        obs.lead_distance,
                        self.params,
                        continuous=self.use_continuous_following,
                    ),
                    0,
                    rule,
                    model,
                )
            return RowCommand(self.state, self.merge_speed, 0, rule, model)

        if self.state == RowState.FOLLOW:
            # (6) Exit mark → leave and stop.
            if obs.exit_mark_seen:
                self.state = RowState.EXIT
                return RowCommand(self.state, cruise * 0.6, 0)
            return RowCommand(
                self.state,
                following_speed(
                    obs.lead_distance,
                    self.params,
                    continuous=self.use_continuous_following,
                ),
                0,
            )

        if self.state == RowState.EXIT:
            # Scripted exit: keep moving briefly, then DONE.
            if not obs.exit_mark_seen and obs.at_yield_line is False:
                # Once the mark leaves the frame and we are off the ring, stop.
                # Callers can also force DONE after a timed scripted turn.
                pass
            # Speed taper; the mux / scripted turn owns steering here.
            return RowCommand(self.state, cruise * 0.4, 0)

        return RowCommand(self.state, 0.0, self.go_votes)

    def finish_exit(self) -> None:
        self.state = RowState.DONE
