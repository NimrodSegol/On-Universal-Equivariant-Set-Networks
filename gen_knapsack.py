import numpy as np
import sys
import os
import utils as utils


def gen_4d(size, train, DIR):
    W = [100, 80, 50]
    set_seize = 50
    datasets50 = np.zeros([size, set_seize, 5])
    for i in range(size):
        rang = np.random.randint(2, 100)
        val1 = np.random.randint(1, rang, [set_seize])
        rang = np.random.randint(2, 25)
        wt1 = np.random.randint(1, rang, [set_seize, 3])
        x1 = knapSack(W, wt1, val1, set_seize)
        datasets50[i, :, 0] = val1
        datasets50[i, :, 1] = wt1[:, 0]
        datasets50[i, :, 2] = wt1[:, 1]
        datasets50[i, :, 3] = wt1[:, 2]
        datasets50[i, :, 4] = x1
    A = utils.make_batch_of_adj_matrices(datasets50[:, :, :4])
    if train:
        name = 'knapsack'+'train'+'.npy'
    else:
        name = 'knapsack' + 'test' + '.npy'
    print(name)
    np.save(os.path.join(DIR, name), datasets50)
    np.save(os.path.join(DIR, 'mat' + name), A)


def knapSack(W, wt, val, n):
    K = np.zeros([n + 1, W[0] + 1, W[1] + 1, W[2] + 1]).astype(np.int)

    # Build table K in bottom
    # up manner
    for i in range(n + 1):
        for w1 in range(W[0] + 1):
            for w2 in range(W[1] + 1):
                for w3 in range(W[2] + 1):
                    if i == 0 or w1 == 0 or w2 == 0 or w3 == 0:
                        pass
                    elif wt[i - 1][0] <= w1 and wt[i - 1][1] <= w2 and wt[i - 1][2] <= w3:
                        K[i][w1][w2][w3] = max(val[i - 1] + K[i - 1][w1 - wt[i-1][0]][w2 - wt[i-1][1]][w3 - wt[i-1][2]],
                                               K[i - 1][w1][w2][w3])
                    else:
                        K[i][w1][w2][w3] = K[i - 1][w1][w2][w3]
            # stores the result of Knapsack
    res = K[n][W[0]][W[1]][W[2]]
    w1, w2, w3 = W[0], W[1], W[2]
    used = [0 for _ in range(n)]
    for i in range(n, 0, -1):
        if res <= 0:
            break
        if res == K[i - 1][w1][w2][w3]:
            continue
        else:
            # This item is included.
            # print(wt[i - 1])
            used[i - 1] = 1

            # Since this weight is included
            # its value is deducted
            res = res - val[i - 1]
            w1 = w1 - wt[i - 1][0]
            w2 = w2 - wt[i - 1][1]
            w3 = w3 - wt[i - 1][2]
    return used


if __name__ == "__main__":
    output_dir = sys.argv[1]
    print(output_dir)
    train_size, test_size = 10000, 1000
    gen_4d(size=train_size, train=True, DIR=output_dir)
    gen_4d(size=test_size, train=False, DIR=output_dir)

