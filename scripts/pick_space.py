from spaces.euclid import Euclid
from spaces.euclid import EuclidDisk
from spaces.sphere import Sphere
from spaces.Matsumoto import Matsumoto
from spaces.Poincare import Poincare
from spaces.car_like_disk import CarLikeDisk
from spaces.car_like_disk2 import CarLikeDisk2
from spaces.car_like_disk3 import CarLikeDisk3
from spaces.panda5 import Panda5
from spaces.obstacle4 import Obstacle4
from spaces.multi_agent import MultiAgent

import torch as th
import numpy as np
import os

norm2 = lambda c: lambda s : c*(s**2).sum(dim=1)
d_norm2 = lambda c: lambda s: 2*c*s
d_gauss = lambda s : -8*th.exp(-4*(s**2).sum(dim=1))[:,None]*s if th.is_tensor(s) else -8*np.exp(-4*(s**2).sum())*s

def load_data(file_path):
    data = np.load(file_path)
    num = data.shape[0]
    return [[th.tensor(data[i][0], dtype = th.float32), th.tensor(data[i][1], dtype = th.float32)] for i in range(num)]

def pick_space(name):
    if name == "Plane":
        return Euclid(2), load_data(os.path.dirname(os.path.realpath(__file__))+ "/../data/Obstacle_eval_states.npy")
    elif name == "Disk":
        return EuclidDisk(2,1.), [[th.tensor([1/2,1/2]), th.tensor([1/2,-1/2])]]
    elif name == "Sphere":
        return Sphere(2), [[th.tensor([1/2,1/2]), th.tensor([1/2,-1/2])]]
    elif name == "Sphere3":
        return Sphere(3), [[th.tensor([1/2,1/2,-1/2]), th.tensor([1/2,1/2,1/2])]]
    elif name == "Poincare":
        return Poincare(2), [[th.tensor([1/2,1/2]), th.tensor([1/2,-1/2])]]
    elif name == "Poincare3":
        return Poincare(3), [[th.tensor([1/2,1/2,-1/2]), th.tensor([1/2,1/2,1/2])]]
    elif name == "CarLikeDisk":
        return CarLikeDisk(), load_data(os.path.dirname(os.path.realpath(__file__))+ "/../data/"+name+"_eval_states.npy")
    elif name == "CarLikeDisk2":
        return CarLikeDisk2(), load_data(os.path.dirname(os.path.realpath(__file__))+ "/../data/CarLikeDisk_eval_states.npy")
    elif name == "CarLikeDisk2-0.2":
        return CarLikeDisk2(r_min = 0.2), load_data(os.path.dirname(os.path.realpath(__file__))+ "/../data/CarLikeDisk_eval_states.npy")
    elif name == "CarLikeDisk3-0.2":
        return CarLikeDisk3(r_min = 0.2), load_data(os.path.dirname(os.path.realpath(__file__))+ "/../data/CarLikeDisk3_eval_states.npy")
    elif name[:10] == "Matsumoto_":
        c = float(name[10:])
        return Matsumoto(dh = d_norm2(c)), load_data(os.path.dirname(os.path.realpath(__file__))+ "/../data/"+name+"_eval_states.npy")
    elif name == "MatsumotoGauss":
        return Matsumoto(dh = d_gauss), load_data(os.path.dirname(os.path.realpath(__file__))+ "/../data/Matsumoto_-1_eval_states.npy")
    elif name == "Panda5":
        return Panda5(wall_x = 0.1), load_data(os.path.dirname(os.path.realpath(__file__))+ "/../data/Panda5_eval_states.npy")
    elif name == "Obstacle4Outer":
        obstacles = th.tensor([[0.2,0.,0.4,0.8],[0.6,0.2,0.8,1.],
                               [-th.inf, -th.inf, th.inf, 0.],[-th.inf, -th.inf, 0., th.inf],
                               [-th.inf, 1., th.inf, th.inf],[1., -th.inf, th.inf, th.inf]])
        return Obstacle4(obstacles), load_data(os.path.dirname(os.path.realpath(__file__))+ "/../data/Obstacle3_eval_states.npy")
    elif name == "MultiAgent":
        return MultiAgent(), load_data(os.path.dirname(os.path.realpath(__file__))+ "/../data/MultiAgent_eval_states.npy")
    elif name == "MultiAgent-3":
        return MultiAgent(N=3), load_data(os.path.dirname(os.path.realpath(__file__))+ "/../data/MultiAgent-3_eval_states.npy")
    elif name == "MultiAgent-3-0.5":
        return MultiAgent(N=3, thres = 0.5), load_data(os.path.dirname(os.path.realpath(__file__))+ "/../data/MultiAgent-3-0.5_eval_states.npy")
    elif name == "MultiAgent-4-0.5":
        return MultiAgent(thres = 0.5), load_data(os.path.dirname(os.path.realpath(__file__))+ "/../data/MultiAgent_eval_states.npy")
    else:
        raise ValueError("Unknown Space")
