# Experiment: PyBullet enviroments
In this experiment we are using our Decision Trasnformer in PyBullet enviroments using the library d4rl-pybullet from this repository. The original idea was to use Mujoco enviroment from gymnasium, but since we had problems on installing the library due to deprecated dependencies and lisence problems, we seek another similar enviroment.

The library d4rl-pybullet has four replicable enviroments: Hopper, Halfcheetah, Ant and Walker2D. Each of these enviroments offers diferents types of dataset as seen in the following table

- `random` denotes datasets sampled with a randomly initialized policy.
- `medium` denotes datasets sampled with a medium-level policy.
- `mixed` denotes datasets collected during policy training.

| id | task | mean reward | std reward | max reward | min reward | samples |
|:-|:-:|:-|:-|:-|:-|:-|
| hopper-bullet-random-v0 | HopperBulletEnv-v0 | 18.64 | 3.04 | 53.21 | -8.58 | 1000000 |
| hopper-bullet-medium-v0 | HopperBulletEnv-v0 | 1078.36 | 325.52 | 1238.9569 | 220.23 | 1000000 |
| hopper-bullet-mixed-v0 | HopperBulletEnv-v0 | 139.08 | 147.62 | 1019.94 | 9.15 | 59345 |
| halfcheetah-bullet-random-v0 | HalfCheetahBulletEnv-v0 | -1304.49 | 99.30 | -945.29 | -1518.58 | 1000000 |
| halfcheetah-bullet-medium-v0 | HalfCheetahBulletEnv-v0 | 787.35 | 104.31 | 844.91 | -522.57 | 1000000 |
| halfcheetah-bullet-mixed-v0 | HalfCheetahBulletEnv-v0 | 453.12 | 498.19 | 801.02 | -1428.22 | 178178 |
| ant-bullet-random-v0 | AntBulletEnv-v0 | 10.35 | 0.31 | 13.04 | 9.82 | 1000000 |
| ant-bullet-medium-v0 | AntBulletEnv-v0 | 570.80 | 104.82 | 816.79 | 70.87 | 1000000 |
| ant-bullet-mixed-v0 | AntBulletEnv-v0 | 255.40 | 196.22 | 609.66 | -32.74 | 53572 |
| walker2d-bullet-random-v0 | Walker2DBulletEnv-v0 | 14.98 | 2.94 | 66.90 | 5.73 | 1000000 |
| walker2d-bullet-medium-v0 | Walker2DBulletEnv-v0 | 1106.68 | 417.79 | 1394.38 | 16.00 | 1000000 |
| walker2d-bullet-mixed-v0 | Walker2DBulletEnv-v0 | 181.51 | 277.71 | 1363.94 | 9.45 | 89772 |
