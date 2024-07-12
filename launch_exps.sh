#!/bin/bash
source /home/yberthel/AVEC/venv/bin/activate

parallel --ssh oarsh --sshloginfile $OAR_FILE_NODES /home/yberthel/AVEC/main.py {1} {2} {3} {4} ::: 0 1 2 3 4 5 6 7 8 9 ::: Hopper-v4 Walker2d-v4 HalfCheetah-v4 Ant-v4 Humanoid-v4 Reacher-v4 ::: PPO AVEC_PPO CORRECTED_AVEC_PPO ::: 1