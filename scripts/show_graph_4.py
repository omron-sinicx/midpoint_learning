import matplotlib.pyplot as plt
import numpy as np

import matplotlib
from matplotlib.ticker import FormatStrFormatter

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams["font.size"] = 20

fig = plt.figure(figsize = (15, 8))

import sys

prob = "Matsumoto_-1"

algos = [("ACDQT-","Our-T", 'solid', 3, 2, 'o'),
         ("ACDQC-","Our-C", 'solid', 3, 2, 'D'),
         ("Seq-", "Seq", 'solid', 5, 1, 'P'),
         ("SGT_", "PG", 'solid', 30, 1, 'x'),
         ("Inter-","Inter", 'solid', 3, 1, 's'),
         ("Alpha2-","2:1", 'solid', 3, 1, 'X'),
         ("Cut-","Cut", 'solid', 3, 1, 'p'),
]         

ax = fig.add_subplot(2,6,(1,3))
N = 10

dire = "../exp/"

for name, label, style, interval, lw, marker in algos:
    interval = (int)(2/3*interval)
    ac_evals = []
    for seed in range(11,11+N):
        s=seed
        ac_evals.append(np.load(dire+prob+"/"+name+str(s)+"/evaluations.npz"))

    leng = min([len(res["timesteps"]) for res in ac_evals])
    timesteps = ac_evals[0]["timesteps"][:leng:interval]
    success_rates = np.array([res["success_rates"][:leng:interval] for res in ac_evals])
    ax.errorbar(timesteps, np.mean(success_rates, axis=0), np.std(success_rates, axis=0)/np.sqrt(N), label = label, linestyle = style, lw = lw, marker = marker)

ax.set_title("Matsumoto")

ax.set_xlabel("Timesteps")
ax.set_ylabel("Success rate")
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

prob = "CarLikeDisk3-0.2"
N=5
ax = fig.add_subplot(2,6,(4,6))

for name, label, style, interval, lw, marker in algos:
    interval = (int)(4/3*interval)
    ac_evals = []
    _prob = prob
    for seed in range(11,16):
        s=seed
        ac_evals.append(np.load(dire+_prob+"/"+name+str(s)+"/evaluations.npz"))
    
    leng = min([len(res["timesteps"]) for res in ac_evals])
    timesteps = ac_evals[0]["timesteps"][:leng:interval]
    success_rates = np.array([res["success_rates"][:leng:interval] for res in ac_evals])
    ax.errorbar(timesteps, np.mean(success_rates, axis=0), np.std(success_rates, axis=0)/np.sqrt(N), label = label, linestyle = style, lw = lw, marker = marker)

ax.set_title("Car-Like")

ax.set_xlabel("Timesteps")
ax.set_ylabel("Success rate")
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

prob = "Obstacle4Outer"

ax = fig.add_subplot(2,6,(7,8))

for name, label, style, interval, lw, marker in algos:
    ac_evals = []
    for seed in range(11,16):
        s=seed
        ac_evals.append(np.load(dire+prob+"/"+name+str(s)+"/evaluations.npz"))
    
    leng = min([len(res["timesteps"]) for res in ac_evals])
    timesteps = ac_evals[0]["timesteps"][:leng:interval]
    success_rates = np.array([res["success_rates"][:leng:interval] for res in ac_evals])
    ax.errorbar(timesteps, np.mean(success_rates, axis=0), np.std(success_rates, axis=0)/np.sqrt(N), label = label, linestyle = style, lw = lw, marker = marker)

ax.set_title("2D Obstacles")

ax.set_xlabel("Timesteps")
ax.set_ylabel("Success rate")
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

prob = "Panda5"

ax = fig.add_subplot(2,6,(9,10))

for name, label, style, interval, lw, marker in algos:
    ac_evals = []
    for seed in range(11,16):
        s=seed
        ac_evals.append(np.load(dire+prob+"/"+name+str(s)+"/evaluations.npz"))
    
    leng = min([len(res["timesteps"]) for res in ac_evals])
    timesteps = ac_evals[0]["timesteps"][:leng:interval]
    success_rates = np.array([res["success_rates"][:leng:interval] for res in ac_evals])
    ax.errorbar(timesteps, np.mean(success_rates, axis=0), np.std(success_rates, axis=0)/np.sqrt(N), label = label, linestyle = style, lw = lw, marker = marker)

ax.set_title("Robotic Arm")

ax.set_xlabel("Timesteps")
ax.set_ylabel("Success rate")
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

prob = "MultiAgent-3-0.5"

ax = fig.add_subplot(2,6,(11,12))

for name, label, style, interval, lw, marker in algos:
    interval*=2
    ac_evals = []
    for seed in range(11,16):
        s=seed
        ac_evals.append(np.load(dire+prob+"/"+name+str(s)+"/evaluations.npz"))
    
    leng = min([len(res["timesteps"]) for res in ac_evals])
    timesteps = ac_evals[0]["timesteps"][:leng:interval]
    success_rates = np.array([res["success_rates"][:leng:interval] for res in ac_evals])
    ax.errorbar(timesteps, np.mean(success_rates, axis=0), np.std(success_rates, axis=0)/np.sqrt(N), label = label, linestyle = style, lw = lw, marker = marker)

ax.set_title("Three Agents")

ax.set_xlabel("Timesteps")
ax.set_ylabel("Success rate")
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

fig.subplots_adjust(left=0.06, right=0.99, bottom=0.18, top=0.95, wspace = 0.7, hspace = 0.5)
plt.legend(bbox_to_anchor = (-0.8,-0.3,), loc = 'upper center', ncol = 7)

plt.savefig("../figures/graphs.pdf")
plt.show()
