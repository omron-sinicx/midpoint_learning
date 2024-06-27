for seed in {11..20}
do
    python3 -m learn --space Matsumoto_-1 --log_dir ../exp/Matsumoto_-1/ACDQT-$seed --total_timesteps 20000000 --depth 6 --eps 0.1 --relation_train --device cuda:0 --timestep_depth_schedule --learning_rate 3e-5 --seed $seed &
    python3 -m learn --space Matsumoto_-1 --log_dir ../exp/Matsumoto_-1/ACDQC-$seed --total_timesteps 20000000 --depth 6 --eps 0.1 --relation_train --device cuda:0 --learning_rate 3e-5 --seed $seed &
    python3 -m learn --space Matsumoto_-1 --log_dir ../exp/Matsumoto_-1/Inter-$seed --total_timesteps 20000000 --depth 6 --eps 0.1 --device cuda:0 --not_midpoint  --learning_rate 3e-5  --seed $seed &
    python3 -m learn --space Matsumoto_-1 --log_dir ../exp/Matsumoto_-1/Alpha2-$seed --total_timesteps 20000000 --depth 6 --eps 0.1 --device cuda:0 --alpha 2.0  --learning_rate 3e-5  --seed $seed &
    python3 -m learn --space Matsumoto_-1 --log_dir ../exp/Matsumoto_-1/Cut-$seed --total_timesteps 20000000 --depth 6 --eps 0.1 --device cuda:0 --not_midpoint --cut_deltas  --learning_rate 3e-5  --seed $seed &
    python3 -m ppo --space Matsumoto_-1 --log_dir ../exp/Matsumoto_-1/Seq-$seed --seed $seed --max_step 64 --total_timesteps 20000000 --device cuda:0 &
    CUDA_VISIBLE_DEVICES=0 python3 -m SGT_PG.sgt_pg_main SGT_PG/config/config_Matsumoto.yml $seed > /dev/null 2> /dev/null &
done
for seed in {11..15}
do
    python3 -m learn --space CarLikeDisk3-0.2 --log_dir ../exp/CarLikeDisk3-0.2/ACDQT-$seed --total_timesteps 80000000 --depth 6 --eps 0.2 --net_arch 400 300 300 --relation_train --device cuda:0 --timestep_depth_schedule --seed $seed &
    python3 -m learn --space CarLikeDisk3-0.2 --log_dir ../exp/CarLikeDisk3-0.2/ACDQC-$seed --total_timesteps 80000000 --depth 6 --eps 0.2 --net_arch 400 300 300 --relation_train --device cuda:0 --seed $seed &
    python3 -m learn --space CarLikeDisk3-0.2 --log_dir ../exp/CarLikeDisk3-0.2/Inter-$seed --total_timesteps 80000000 --depth 6 --eps 0.2 --net_arch 400 300 300 --device cuda:0 --not_midpoint --seed $seed &
    python3 -m learn --space CarLikeDisk3-0.2 --log_dir ../exp/CarLikeDisk3-0.2/Alpha2-$seed --total_timesteps 80000000 --depth 6 --eps 0.2 --net_arch 400 300 300 --device cuda:0 --alpha 2.0 --seed $seed &
    python3 -m learn --space CarLikeDisk3-0.2 --log_dir ../exp/CarLikeDisk3-0.2/Cut-$seed --total_timesteps 80000000 --depth 6 --eps 0.2 --net_arch 400 300 300 --device cuda:0 --not_midpoint --cut_deltas --seed $seed &
    python3 -m ppo --space CarLikeDisk3-0.2 --log_dir ../exp/CarLikeDisk3-0.2/Seq-$seed --seed $seed --max_step 64 --eps 0.2 --total_timesteps 80000000 --device cuda:0 --net_arch 400 300 300 --learning_rate 3e-4 &
    CUDA_VISIBLE_DEVICES=0 python3 -m SGT_PG.sgt_pg_main SGT_PG/config/config_CarLikeDisk3-0.2.yml $seed > /dev/null 2> /dev/null &
    python3 -m learn --space Obstacle4Outer --log_dir ../exp/Obstacle4Outer/ACDQT-$seed --total_timesteps 40000000 --depth 6 --eps 0.1 --net_arch 400 300 300 --relation_train --device cuda:0 --timestep_depth_schedule --seed $seed &
    python3 -m learn --space Obstacle4Outer --log_dir ../exp/Obstacle4Outer/ACDQC-$seed --total_timesteps 40000000 --depth 6 --eps 0.1 --net_arch 400 300 300 --relation_train --device cuda:0 --seed $seed &
    python3 -m learn --space Obstacle4Outer --log_dir ../exp/Obstacle4Outer/Inter-$seed --total_timesteps 40000000 --depth 6 --eps 0.1 --net_arch 400 300 300 --device cuda:0 --timestep_depth_schedule --not_midpoint --seed $seed &
    python3 -m learn --space Obstacle4Outer --log_dir ../exp/Obstacle4Outer/Alpha2-$seed --total_timesteps 40000000 --depth 6 --eps 0.1 --net_arch 400 300 300 --device cuda:0 --timestep_depth_schedule --alpha 2.0 --seed $seed &
    python3 -m learn --space Obstacle4Outer --log_dir ../exp/Obstacle4Outer/Cut-$seed --total_timesteps 40000000 --depth 6 --eps 0.1 --net_arch 400 300 300 --device cuda:0 --timestep_depth_schedule --not_midpoint --cut_deltas --seed $seed &
    python3 -m ppo --space Obstacle4Outer --log_dir ../exp/Obstacle4Outer/Seq-$seed --seed $seed --max_step 64 --total_timesteps 40000000 --eps 0.1 --net_arch 400 300 300 --device cuda:0 --learning_rate 3e-4 &
    CUDA_VISIBLE_DEVICES=0 python3 -m SGT_PG.sgt_pg_main SGT_PG/config/config_Obstacle4Outer.yml $seed > /dev/null 2> /dev/null &
    python3 -m learn --space Panda5 --log_dir ../exp/Panda5/ACDQT-$seed --total_timesteps 40000000 --depth 6 --eps 0.2 --net_arch 400 300 300 --relation_train --device cuda:0 --timestep_depth_schedule --seed $seed &
    python3 -m learn --space Panda5 --log_dir ../exp/Panda5/ACDQC-$seed --total_timesteps 40000000 --depth 6 --eps 0.2 --net_arch 400 300 300 --relation_train --device cuda:0 --seed $seed &
    python3 -m learn --space Panda5 --log_dir ../exp/Panda5/Inter-$seed --total_timesteps 40000000 --depth 6 --eps 0.2 --net_arch 400 300 300 --device cuda:0 --timestep_depth_schedule --not_midpoint --seed $seed &
    python3 -m learn --space Panda5 --log_dir ../exp/Panda5/Alpha2-$seed --total_timesteps 40000000 --depth 6 --eps 0.2 --net_arch 400 300 300 --device cuda:0 --timestep_depth_schedule --alpha 2.0 --seed $seed &
    python3 -m learn --space Panda5 --log_dir ../exp/Panda5/Cut-$seed --total_timesteps 40000000 --depth 6 --eps 0.2 --net_arch 400 300 300 --device cuda:0 --timestep_depth_schedule --not_midpoint --cut_deltas --seed $seed &
    python3 -m ppo --space Panda5 --log_dir ../exp/Panda5/Seq-$seed --seed $seed --max_step 64 --total_timesteps 40000000 --eps 0.2 --net_arch 400 300 300 --device cuda:0 --learning_rate 3e-4 &
    CUDA_VISIBLE_DEVICES=0 python3 -m SGT_PG.sgt_pg_main SGT_PG/config/config_Panda5.yml $seed > /dev/null 2> /dev/null &
    python3 -m learn --space MultiAgent-3-0.5 --log_dir ../exp/MultiAgent-3-0.5/ACDQT-$seed --total_timesteps 80000000 --depth 6 --eps 0.2 --net_arch 400 300 300 --relation_train --device cuda:0 --timestep_depth_schedule --seed $seed &
    python3 -m learn --space MultiAgent-3-0.5 --log_dir ../exp/MultiAgent-3-0.5/ACDQC-$seed --total_timesteps 80000000 --depth 6 --eps 0.2 --net_arch 400 300 300 --relation_train --device cuda:0 --seed $seed &
    python3 -m learn --space MultiAgent-3-0.5 --log_dir ../exp/MultiAgent-3-0.5/Inter-$seed --total_timesteps 80000000 --depth 6 --eps 0.2 --net_arch 400 300 300 --device cuda:0 --timestep_depth_schedule --not_midpoint --seed $seed &
    python3 -m learn --space MultiAgent-3-0.5 --log_dir ../exp/MultiAgent-3-0.5/Alpha2-$seed --total_timesteps 80000000 --depth 6 --eps 0.2 --net_arch 400 300 300 --device cuda:0 --timestep_depth_schedule --alpha 2.0 --seed $seed &
    python3 -m learn --space MultiAgent-3-0.5 --log_dir ../exp/MultiAgent-3-0.5/Cut-$seed --total_timesteps 80000000 --depth 6 --eps 0.2 --net_arch 400 300 300 --device cuda:0 --timestep_depth_schedule --not_midpoint --cut_deltas --seed $seed &
    python3 -m ppo --space MultiAgent-3-0.5 --log_dir ../exp/MultiAgent-3-0.5/Seq-$seed --seed $seed --max_step 64 --total_timesteps 80000000 --eps 0.2 --net_arch 400 300 300 --device cuda:0 --learning_rate 3e-4 &
    CUDA_VISIBLE_DEVICES=0 python3 -m SGT_PG.sgt_pg_main SGT_PG/config/config_MultiAgent-3-0.5.yml $seed > /dev/null 2> /dev/null &
done
