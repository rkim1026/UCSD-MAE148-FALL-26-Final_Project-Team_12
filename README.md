# MAE/ECE 148 · Team 12 — Roundabout Right-of-Way Negotiation

ROS 2 implementation of the modular stack from the [Roundabout Build Guide](https://claude.ai/artifact/514uge8RnSLbgM5JtoJwHf): camera lane following + LiDAR traffic tracking + a five-state yield / merge / follow / exit machine. Steering and speed are separate channels (DonkeyCar “Local Angle” style). A hand-written gap rule is the must-have; an optional go/wait classifier can only **veto** a go.

```
Camera ──► Lane follower ──────────────────────► steering
                                              ╲
LiDAR ──► Car tracker ──► Ring state ──► ROW logic ──► speed ──► /cmd_vel → VESC
                              ▲
Camera ──► Yield / exit tape ─┘
```

## Gap rule

\[
\text{gap time} = R \cdot \varphi / v
\qquad
\text{go} \iff \text{zone empty} \land \text{gap time} > T_{\text{clear}} + \text{margin}
\]

Measure \(T_{\text{clear}}\) on the car (slowest of 10 standstill-to-clear runs). Start with `margin = 0.5 s`.

## Package layout

```
src/roundabout_pkg/
  roundabout_pkg/
    ring_gap.py          # geometry + gap rule (shared with sim)
    car_tracker.py       # LaserScan → φ, speed, lead distance
    state_machine.py     # APPROACH → YIELD → MERGE → FOLLOW → EXIT
    lane_follower.py     # OpenCV + feedforward atan(L/R) + PID
    yield_detector.py    # orange yield line / exit mark
    nodes/               # ROS 2 entry points
    sim/scenario_sim.py  # imports the real decision code
    ml/                  # hindsight labels + logistic veto
  config/roundabout.yaml
  launch/roundabout.launch.py
  test/test_ring_gap.py
```

## Build (UCSD Robocar Docker / Jetson)

Copy this repo’s `src/roundabout_pkg` into the Robocar workspace (or clone this repo as the workspace root):

```bash
# inside the UCSD Robocar devel image
source_ros2          # or: source /opt/ros/$ROS_DISTRO/setup.bash
cd ~/projects/ros2_ws   # adjust to your workspace
# place roundabout_pkg under src/
colcon build --packages-select roundabout_pkg
source install/setup.bash
```

Optional pip deps for ML / desk tests: `numpy`, `opencv-python`, `scikit-learn`.

### Hook into `all_nodes.launch.py`

In `ucsd_robocar_nav2_pkg/config/node_pkg_locations_ucsd.yaml`:

```yaml
roundabout_pkg: ["roundabout_pkg", "roundabout.launch.py"]
```

In `node_config.yaml`:

```yaml
roundabout_pkg: 1
```

In `car_config.yaml`, enable your LiDAR, camera (OAK-D / intel), and `vesc`.

Or launch the stack alone once sensors + VESC are already running:

```bash
ros2 launch roundabout_pkg roundabout.launch.py
```

Hold-to-run: publish `true` on `/joy_enable` (or set `require_enable: false` in `config/roundabout.yaml` for desk tests).

## Topics

| Topic | Type | Role |
| --- | --- | --- |
| `/camera/color/image_raw` | `sensor_msgs/Image` | Lane + tape detection |
| `/scan` | `sensor_msgs/LaserScan` | Circulating-car tracker |
| `/lane_steering` | `std_msgs/Float32` | Steering in \([-1,1]\) |
| `/at_yield_line` | `std_msgs/Bool` | Orange tape at stop row |
| `/exit_mark` | `std_msgs/Bool` | Exit tape visible |
| `/ring_state` | `Float32MultiArray` | `[φ, speed, gap_time, lead, busy, front_clear]` |
| `/row_speed` | `std_msgs/Float32` | Desired speed (m/s) |
| `/row_state` | `std_msgs/String` | State machine name |
| `/cmd_vel` | `geometry_msgs/Twist` | VESC (`linear.x` throttle, `angular.z` steer) |

Record everything from day one:

```bash
ros2 bag record /scan /camera/color/image_raw /ring_state /row_state /row_speed /lane_steering /cmd_vel
```

## Desk test (no car)

```bash
# from repo root, no ROS needed for core + sim
PYTHONPATH=src/roundabout_pkg python3 -m pytest src/roundabout_pkg/test -q
PYTHONPATH=src/roundabout_pkg python3 -m roundabout_pkg.sim.scenario_sim
```

The scenario sim **imports** `ring_gap.py` / `state_machine.py`, so the code you batch-test is the code that drives the car.

## Floor checklist (from the build guide)

1. **Track** — lane ~0.7 m wide, centre radius ~1.3 m, flat tape-only island, orange yield line, third colour at exit. Cardboard band on the human car at LiDAR height.
2. **Lane follow** — 10 clean laps; add feedforward `atan(wheelbase / R)`.
3. **Tracker** — car cluster visible in a plotted scan; speed within ~10% of stopwatch.
4. **Yield** — measure \(T_{\text{clear}}\); 20 timed entries, zero collisions.
5. **Follow + exit** — never close under 0.4 m; exit 10/10.
6. **Test matrix** — human slow/fast × early/on-time/late arrival × stop on ring.

## Optional ML veto

```python
# after collecting wait scans + future trajectories from bags
from roundabout_pkg.ml.go_wait_classifier import train_from_arrays
clf = train_from_arrays(X, y, out_path="go_wait.json")
```

Set in `roundabout.yaml`:

```yaml
use_classifier_veto: true
classifier_path: "/home/projects/go_wait.json"
```

Final decision is always `rule_go AND model_go`.

## References

- Team build guide: Roundabout Right-of-Way Negotiation, end to end
- [DonkeySim / DonkeyCar class docs](https://ucsd-silberman-classes-and-projects.github.io/Markdown-Instructions/donkeysim.html#s3)
- UCSD Robocar hub (`ucsd_robocar_hub2`) — `/scan`, `/cmd_vel` conventions
