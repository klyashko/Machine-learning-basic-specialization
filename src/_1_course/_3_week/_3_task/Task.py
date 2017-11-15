import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from src.utils.Utils import save_answer


def f(x):
    return np.sin(x / 5.) * np.exp(x / 10.) + 5. * np.exp(-x / 2.)


def h(x):
    return int(f(x))

x = range(1, 31)
y = [h(i) for i in x]

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

result = minimize(h, 30, method='BFGS')
print('Iterations', result.nfev)
first_answer = round(result.fun, 2)
print()

measurement = differential_evolution(h, [(1, 30)])
print('Iterations', measurement.nfev)

second_answer = round(measurement.fun, 2)

save_answer(3, [first_answer, second_answer], space=True)
