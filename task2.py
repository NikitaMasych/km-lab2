import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

x_data = np.array([-1.6000, -1.2000, -0.8000, -0.4000, 0, 0.4000, 0.8000, 1.2000, 1.6000, 2.0000])

y_data_linear = np.array([-0.2000, 0.6000, 1.4000, 2.2000, 3.0000, 3.8000, 4.6000, 5.4000, 6.2000, 7.0000])
y_data_quadratic = np.array([4.3200, 3.2800, 2.8800, 3.1200, 4.0000, 5.5200, 7.6800, 10.4800, 13.9200, 18.0000])

k2 = 7
k1 = k2 * 0.01

x_data += k2

y_data_linear += k1
y_data_quadratic += k1

def linear_func(x, A, B):
    return A * x + B

def quadratic_func(x, A, B, C):
    return A * x**2 + B * x + C

params_linear, covariance_linear = curve_fit(linear_func, x_data, y_data_linear)
A_linear, B_linear = params_linear

params_quadratic, covariance_quadratic = curve_fit(quadratic_func, x_data, y_data_quadratic)
A_quadratic, B_quadratic, C_quadratic = params_quadratic

x_vals = np.linspace(min(x_data), max(x_data), 100)

y_linear = linear_func(x_vals, A_linear, B_linear)
y_quadratic = quadratic_func(x_vals, A_quadratic, B_quadratic, C_quadratic)

print("Лінійна апроксимація: y = {:.4f} * x + {:.4f}".format(A_linear, B_linear))
print("Квадратична апроксимація: y = {:.4f} * x^2 + {:.4f} * x + {:.4f}".format(A_quadratic, B_quadratic, C_quadratic))

plt.scatter(x_data, y_data_linear, label='Лінійні дані', color='black')
plt.scatter(x_data, y_data_quadratic, label='Квадратичні дані', color='red')

print(x_data)
print(y_data_linear)

plt.plot(x_vals, y_linear, label='Лінійна апроксимація', linestyle='dashed', color='black')
plt.plot(x_vals, y_quadratic, label='Квадратична апроксимація', linestyle='dashed', color='red')

plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Метод найменших квадратів')
plt.show()