import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from src.utils.Utils import save_answer


def f(x):
    return np.sin(x / 5.) * np.exp(x / 10.) + 5. * np.exp(-x / 2.)


def measurement(x0):
    result = minimize(f, x0, method='BFGS')
    print('Iterations', result.nfev)
    # return round(result.x[0], 2)
    return round(result.fun, 2)


x = range(1, 31)
y = [f(i) for i in x]

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

for i in range(1, 31):
    print('Iteration', i)
    # print(minimize(f, i).x)
    print(minimize(f, i).fun)
    print()
    print()

save_answer(1, [measurement(2), measurement(30)], space=True)
