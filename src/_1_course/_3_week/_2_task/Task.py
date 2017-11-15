import numpy as np

from scipy.optimize import differential_evolution
from src.utils.Utils import save_answer


def f(x):
    return np.sin(x / 5.) * np.exp(x / 10.) + 5. * np.exp(-x / 2.)


measurement = differential_evolution(f, [(1, 30)])
print('Iterations', measurement.nfev)
# print(measurement.fun[0])

answer = round(measurement.fun[0], 2)

save_answer(2, answer)
