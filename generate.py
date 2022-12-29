import numpy as np

seeds = np.random.randint(1,1000000)
choices = np.random.choice(seeds, 100)

for i in choices:
    print("python run.py --seed {}".format(i))