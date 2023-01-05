import argparse
import numpy as np
import os
import ast
import scipy
from helpers import *
# import yaml
import time
import pickle

if __name__ == "__main__":
    
    X = [-3,0,3]
    Q = len(X)
    N = 50
    search_interval = [-8,8]
    int_start = search_interval[0]
    int_end = search_interval[1]
    step = (int_end-int_start)/N
    S = np.linspace(int_start, int_end, N+1)
    M = Q

    sigma = 1
    Y = X + np.random.randn(Q)*sigma
    Phi = [scipy.stats.norm(loc=X[i], scale=sigma) for i in range(Q)]
    
    print("loaded config")
    print("X={}, N={}, M={}, Sigma={}".format(X, N, M, sigma))
    print("start={}, end={}".format(int_start, int_end))

    log_fn = "logs/exhaustive_search/3level.time{}.log".format(int(time.time()))
    with open(log_fn, "w") as f:
        f.writelines("X={}, N={}, M={}, Sigma={}\n".format(X, N, M, sigma))
        f.writelines("start={}, end={}\n".format(int_start, int_end))
        f.writelines("\n")

    start_ex = time.time()
    search_thresh = S[1:-1]
    print(search_thresh)
    data = {}
    current_best = -np.inf
    current_best_thres = None
    for idx_h1, h1 in tqdm.tqdm(enumerate(search_thresh[:-1])):
        start = time.time()
        for idx_h2, h2 in enumerate(search_thresh[idx_h1+1:]):
            Hx = [S[0]] + [h1, h2] + [S[-1]]
            # print(Hx)
            Azx = np.zeros((M, Q))
            for j in range(Q):
                for i in range(M):
                    Azx[i, j] = Phi[j].cdf(Hx[i+1]) - Phi[j].cdf(Hx[i])
            status, obj_value, px_value = calculate_optimal_distribution(Q, M, Azx)
            if status == "optimal":
                Px = px_value
                Ayx, Axy, Py, Pxy = calculate_transition_matrix(Px, N, Q, S, Phi)
                data[(h1, h2)] = calculate_I(Px, Q, M, Phi, [S[0], h1, h2, S[-1]])
            if data[(h1, h2)] > current_best:
                current_best = data[(h1, h2)]
                current_best_thres = (h1, h2)
                current_best_thres_dist = Px
        stop = time.time()

        opt_H = np.array([S[0]] + list(current_best_thres) + [S[-1]])
        print("iter {}".format(idx_h1))
        print("    optimal quantizer {}".format(opt_H))
        print("    optimal input distribution {}".format(current_best_thres_dist))
        print("    optimal I(X;Z) {}".format(current_best))
        print("    took {:.4f}s".format(stop-start))
        with open(log_fn, "a") as f:
            f.writelines("iter {}\n".format(idx_h1))
            f.writelines("    optimal quantizer {}\n".format(opt_H))
            f.writelines("    optimal input distribution {}\n".format(current_best_thres_dist))
            f.writelines("    optimal I(X;Z) {}\n".format(current_best))
            f.writelines("    took {:.4f}s\n".format(stop-start))
        # if (idx_h1+1)%10 == 0:
        #     with open("data_5lv_exhaustive_data_N{}.pickle".format(idx_h1, N), "wb") as f:
        #         pickle.dump(data, f)
    with open(log_fn, "a") as f:
        f.writelines("Took {}s\n".format(time.time() - start_ex))
        f.writelines("End solving by exhaustive search\n\n")