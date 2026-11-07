\# TwinDrive

Lane-free autonomous driving on a digital twin of an Indian university campus.

TwinDrive is an autonomous driving stack built in CARLA on a 1:1 digital twin of
the SRM Institute of Science and Technology, Kattankulathur campus. Most CARLA
maps have clean, well-marked roads. Campus roads in India are irregular, have
little lane discipline, and carry a mix of cars, two-wheelers and pedestrians
crossing anywhere. The goal of this project is a driving agent that reasons
about drivable space rather than lane centre lines, trained with imitation
learning followed by reinforcement learning.

## Status

Work in progress, built phase by phase.

| Phase | Scope | Status |
|---|---|---|
| 0 | Map audit, road graph, drivable-area mask, campus landmarks | in progress |
| 1 | Ego sensors (3D LiDAR, cameras), ROS 2 visualisation, route following | planned |
| 2 | Campus traffic module (two-wheelers, crowds, parked vehicles) | planned |
| 3 | LiDAR + camera bird's-eye-view perception | planned |
| 4 | Imitation learning from human driving demonstrations | planned |
| 5 | Reinforcement learning fine-tuning with a lane-free reward | planned |
| 6 | Evaluation suite and benchmark | planned |

## Setup

The simulator runs on Windows (packaged CARLA 0.9.16 build with the campus
map) and all project code runs in WSL2 Ubuntu 22.04, connected over localhost.

```bash
# WSL2: enable mirrored networking in %USERPROFILE%\.wslconfig
#   [wsl2]
#   networkingMode=mirrored

conda create -n twindrive python=3.10 -y
conda activate twindrive
pip install -e .
```

## Layout

```
twindrive/          core package (no ROS dependency)
  world/            road graph and routability analysis
tools/              standalone scripts, numbered in the order they are used
config/             campus landmarks used as route goals
maps/cm2/           generated drivable-area mask metadata
```

## Tools

| Script | Purpose |
|---|---|
| `tools/00_connect.py` | Connection and map audit |
| `tools/01_compare_maps.py` | Routability audit (strongly connected components, dead ends) |
| `tools/02_build_drivable_mask.py` | Global lane-free drivable-area mask |
| `tools/03_landmarks.py` | Name campus landmarks as route goals |


