# Generation of Geodesics with Actor-Critic Reinforcement Learning to Predict Midpoints

This is a repository for [the following paper](https://arxiv.org/abs/2407.01991):
- Kazumi Kasaura. 2024. “Generation of Geodesics with Actor-Critic Reinforcement Learning to Predict Midpoints”

It contains all scripts and a dockerfile to reproduce our experiments.

Contents in [scripts/SGT_PG/](scripts/SGT_PG/) are modified contents in https://github.com/tomjur/SGT-PG.

## Experiments

### Building an environment

You can use [Dockerfile](Dockerfile) to build an environment.

### Running

[scripts/learn.py](scripts/learn.py) is the main script for our proposed methods. You can specify the environment, method variants, and hyperparameters with arguments.

[scripts/ppo.py](scripts/ppo.py) is the main script for the sequential reinforcement learning baseline.

[scripts/SGT_PG/sgt_pg_main.py](scripts/SGT_PG/sgt_pg_main.py) is the main script for the policy gradient baseline.

You can use a script [scripts/run.sh](scripts/run.sh) to run all experiments.

The IDs of used GPU can be specified by editting this script.

The results are stored in `exp` folder.

```
mkdir exp
cd scripts
bash run.sh
cd ..
```

### Viewing Results

After all experiments are done, results can be plotted by a script [scripts/show_graph_4.py](scripts/show_graph_4.py).

Examples of generated paths can be visualized by scripts.
The images are stored in `figures` folder.

Calculation of lengths of generated paths can be done by a script [scripts/compare_cost.py](scripts/compare_cost.py).
The result is stored in `data` folder and the comparison table can be made by a script [scripts/make_table.py](scripts/make_table.py).

```
mkdir figures
cd scripts
python3 show_graph_4.py
python3 view_trajs.py
python3 view_trajs_car3.py
python3 view_traj_obstacles.py
python3 visualize_panda.py
python3 visualize_multiagents.py
python3 compare_cost.py
python3 make_table.py
cd ..
```

## License
This software is released under the MIT License, see [LICENSE](LICENSE).

## Citation
```
@article{kasaura2024generation,
  title={Generation of Geodesics with Actor-Critic Reinforcement Learning to Predict Midpoints},
  author={Kasaura, Kazumi},
  journal={arXiv preprint arXiv:2407.01991},
  year={2024}
}
```
