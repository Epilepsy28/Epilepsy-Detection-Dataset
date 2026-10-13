import numpy as np
from matplotlib import pyplot as plt


# Sample Signal
def Sample_Signal():
    for n in range(1):
        sig = np.load('Signal.npy', allow_pickle=True).astype(np.float32)
        siganl = np.asarray(sig)
        for j in range(5):
            signals = siganl[j]
            fig = plt.figure()
            ax = fig.add_axes([0.22, 0.22, 0.7, 0.7])
            ax.plot(signals)
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.show()


if __name__ == '__main__':
    Sample_Signal()
