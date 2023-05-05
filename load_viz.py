import pickle as pkl
import matplotlib
import matplotlib.pyplot as plt
import os

matplotlib.use('TkAgg')

if __name__ == '__main__':
    files = ['experiments/' + fn
             for fn in os.listdir('experiments')
             if fn.endswith('3d')]
    for fn in files:
        with open(fn, 'rb') as handle:
            data = pkl.load(handle)
            fig, axes = data['fig'], data['axes']
            plt.show()
    while True:
        pass
