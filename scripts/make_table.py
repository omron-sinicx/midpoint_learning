import numpy as np

probs = [("Matsumoto_-1", 6, 10, [64, 64], 0.1),
         ("CarLikeDisk3-0.2", 6, 5, [400, 300, 300], 0.2),
         ("Obstacle4Outer", 6, 5, [400, 300, 300], 0.1),
         ("Panda5", 6, 5, [400, 300, 300], 0.2),
         ("MultiAgent-3-0.5", 6, 5, [400, 300, 300], 0.2)
]

algos = [("ACDQT-","Our-T"),
         ("ACDQC-","Our-C"),
         ("Seq-", "Seq"),
         ("Inter-","Inter"),
         ("Alpha2-","2:1"),
         ("Cut-","Cut"),
]         

for prob, depth, N, net_arch, eps in probs:
    ress = np.load("../data/"+prob+"_compare.npy")
    print(prob)
    valid_alogs = []
    for i in range(len(algos)):
        if ress[:,i,:,0].sum() >= N * 5:
            valid_alogs.append((i, algos[i][1]))
            print('&', algos[i][1], end = "")
    print("\\\\\n\\hline")
    for i, name_i in valid_alogs:
        print(name_i, end = "")
        for j, name_j in valid_alogs:
            if i == j:
                print("&-",end="")
                continue
            rates = []
            percs = []
            for seed in range(N):
                win_c = 0
                all_c = 0
                for k in range(100):
                    if ress[seed][i][k][0] and ress[seed][j][k][0]:
                        all_c+=1
                        if ress[seed][i][k][1] > ress[seed][j][k][1]:
                            win_c+=1
                if all_c > 0:
                    rates.append(100*win_c/all_c)
                percs.append(100*all_c/100)
            rates = np.array(rates)
            percs = np.array(percs)
            print(f"& ${np.mean(rates):.0f} \pm {np.std(rates)/np.sqrt(len(rates)):.0f}$ (${np.mean(percs):.0f}$)", end = "")
        print("\\\\")
