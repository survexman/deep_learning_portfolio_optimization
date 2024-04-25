import numpy as np


def sin_single(n_timesteps, amplitude = 1, freq = 0.25, phase = 0):
    x = np.arange(n_timesteps)
    return amplitude * np.sin(2 * np.pi * freq * x + phase)
