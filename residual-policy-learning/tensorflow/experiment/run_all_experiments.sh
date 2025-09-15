#!/bin/bash
envs=("FetchPush-v1" "FetchPickAndPlace-v1" "SlipperyPush-v0" "TwoFrameHookNoisy-v0" "ComplexHook-v0" "OtherPusherEnv-v0" "TwoFrameNoisyOtherPusherEnv-v0")
residuals=("ResidualMPCPush-v0" "ResidualFetchPickAndPlace-v0" "ResidualSlipperyPush-v0" "TwoFrameResidualHookNoisy-v0" "ResidualComplexHook-v0" "ResidualOtherPusherEnv-v0" "ResidualTwoFramePusherNoisyEnv-v0")
expertexplore=("MPCPush-v0" "FetchPickAndPlace-v1" "SlipperyPush-v0" "TwoFrameHookNoisy-v0" "ComplexHook-v0" "OtherPusherEnv-v0" "TwoFrameNoisyOtherPusherEnv-v0")

eps=(0.3 0.3 0.3 0.6 0.6 0.3 0.3)
alpha=(0.8 0.8 0.8 0.8 0.8 0.8 0.8)
configs=("push.json" "pickandplace.json" "push.json" "hook.json" "hook.json" "push.json" "push.json")
cpus=(19 19 19 1 1 1 1)
nepochs=(50 50 50 300 300 300 300)
seeds=(0 1 2 3 4)

for j in ${!seeds[@]}; do
    #Train expert-explore
    for i in ${!expertexplore[@]}; do
        python train_staged.py --env ${expertexplore[$i]} --n_epochs ${nepochs[$i]} --num_cpu ${cpus[$i]} --config_path=configs/${configs[$i]} --logdir ./logs/seed${seeds[$j]}/${expertexplore[$i]}_expertexplore --seed ${seeds[$j]} --random_eps=${eps[$i]} --controller_prop=${alpha[$i]}
    done

    #Train from scratch
    for i in ${!envs[@]}; do
        python train_staged.py --env ${envs[$i]} --n_epochs ${nepochs[$i]} --num_cpu ${cpus[$i]} --config_path=configs/${configs[$i]} --logdir ./logs/seed${seeds[$j]}/${envs[$i]} --seed ${seeds[$j]}
    done

    #Train residuals
    for i in ${!residuals[@]}; do
        python train_staged.py --env ${residuals[$i]} --n_epochs ${nepochs[$i]} --num_cpu ${cpus[$i]} --config_path=configs/${configs[$i]} --logdir ./logs/seed${seeds[$j]}/${residuals[$i]} --seed ${seeds[$j]}
    done
done


# """
cd residual-policy-learning/tensorflow/experiment/

# # Expert Explore
# python train_staged.py --env MPCPush-v0 --n_epochs 50 --num_cpu 19 --config_path=configs/push.json --logdir ./logs/seed0/MPCPush-v0_expertexplore --seed 0 --random_eps=0.3 --controller_prop=0.8
# python train_staged.py --env FetchPickAndPlace-v1 --n_epochs 50 --num_cpu 19 --config_path=configs/pickandplace.json --logdir ./logs/seed0/FetchPickAndPlace-v1_expertexplore --seed 0 --random_eps=0.3 --controller_prop=0.8
# python train_staged.py --env SlipperyPush-v0 --n_epochs 50 --num_cpu 19 --config_path=configs/push.json --logdir ./logs/seed0/SlipperyPush-v0_expertexplore --seed 0 --random_eps=0.3 --controller_prop=0.8
# python train_staged.py --env TwoFrameHookNoisy-v0 --n_epochs 300 --num_cpu 1 --config_path=configs/hook.json --logdir ./logs/seed0/TwoFrameHookNoisy-v0_expertexplore --seed 0 --random_eps=0.6 --controller_prop=0.8
python train_staged.py --env ComplexHook-v0 --n_epochs 300 --num_cpu 1 --config_path=configs/hook.json --logdir ./logs/seed0/ComplexHook-v0_expertexplore --seed 0 --random_eps=0.6 --controller_prop=0.8
# python train_staged.py --env OtherPusherEnv-v0 --n_epochs 300 --num_cpu 1 --config_path=configs/push.json --logdir ./logs/seed0/OtherPusherEnv-v0_expertexplore --seed 0 --random_eps=0.3 --controller_prop=0.8
# 

# # From Scratch
# python train_staged.py --env FetchPush-v1 --n_epochs 50 --num_cpu 19 --config_path=configs/push.json --logdir ./logs/seed0/FetchPush-v1 --seed 0
# python train_staged.py --env FetchPickAndPlace-v1 --n_epochs 50 --num_cpu 19 --config_path=configs/pickandplace.json --logdir ./logs/seed0/FetchPickAndPlace-v1 --seed 0
# python train_staged.py --env SlipperyPush-v0 --n_epochs 50 --num_cpu 19 --config_path=configs/push.json --logdir ./logs/seed0/SlipperyPush-v0 --seed 0
# python train_staged.py --env TwoFrameHookNoisy-v0 --n_epochs 300 --num_cpu 1 --config_path=configs/hook.json --logdir ./logs/seed0/TwoFrameHookNoisy-v0 --seed 0
python train_staged.py --env ComplexHook-v0 --n_epochs 300 --num_cpu 1 --config_path=configs/hook.json --logdir ./logs/seed0/ComplexHook-v0 --seed 0
# python train_staged.py --env OtherPusherEnv-v0 --n_epochs 300 --num_cpu 1 --config_path=configs/push.json --logdir ./logs/seed0/OtherPusherEnv-v0 --seed 0
# 

# # Residual
# python train_staged.py --env ResidualMPCPush-v0 --n_epochs 50 --num_cpu 19 --config_path=configs/push.json --logdir ./logs/seed0/ResidualMPCPush-v0 --seed 0
# python train_staged.py --env ResidualFetchPickAndPlace-v0 --n_epochs 50 --num_cpu 19 --config_path=configs/pickandplace.json --logdir ./logs/seed0/ResidualFetchPickAndPlace-v0 --seed 0
# python train_staged.py --env ResidualSlipperyPush-v0 --n_epochs 50 --num_cpu 19 --config_path=configs/push.json --logdir ./logs/seed0/ResidualSlipperyPush-v0 --seed 0
# python train_staged.py --env TwoFrameResidualHookNoisy-v0 --n_epochs 300 --num_cpu 1 --config_path=configs/hook.json --logdir ./logs/seed0/TwoFrameResidualHookNoisy-v0 --seed 0
python train_staged.py --env ResidualComplexHook-v0 --n_epochs 300 --num_cpu 1 --config_path=configs/hook.json --logdir ./logs/seed0/ResidualComplexHook-v0 --seed 0
# python train_staged.py --env ResidualOtherPusherEnv-v0 --n_epochs 300 --num_cpu 1 --config_path=configs/push.json --logdir ./logs/seed0/ResidualOtherPusherEnv-v0 --seed 0

# """
