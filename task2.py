import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Вхідні дані (вузли інтерполяції)
x_data = np.array([8.6, 5.8, 6.2, 6.6, 7, 7.4, 7.8, 8.2, 8.6, 9])

# Різні значення y_data для лінійної і квадратичної функцій
y_data_linear = np.array([-0.13, 0.67, 1.47, 2.27, 3.07, 3.87, 4.67, 5.47, 6.27, 7.07])
y_data_quadratic = np.array([4.39, 3.35, 2.95, 3.19, 4.07, 5.59, 7.75, 10.55, 13.99, 18.07])

# Лінійна функція для апроксимації: y = A*x + B
def linear_func(x, A, B):
    return A * x + B

# Квадратична функція для апроксимації: y = A*x^2 + B*x + C
def quadratic_func(x, A, B, C):
    return A * x**2 + B * x + C

# Апроксимація лінійною функцією
params_linear, covariance_linear = curve_fit(linear_func, x_data, y_data_linear)
A_linear, B_linear = params_linear

# Апроксимація квадратичною функцією
params_quadratic, covariance_quadratic = curve_fit(quadratic_func, x_data, y_data_quadratic)
A_quadratic, B_quadratic, C_quadratic = params_quadratic

# Генерація значень для побудови кривих
x_vals = np.linspace(min(x_data), max(x_data), 100)
y_linear = linear_func(x_vals, A_linear, B_linear)
y_quadratic = quadratic_func(x_vals, A_quadratic, B_quadratic, C_quadratic)

# Виведення результатів
print("Лінійна апроксимація: y = {:.4f} * x + {:.4f}".format(A_linear, B_linear))
print("Квадратична апроксимація: y = {:.4f} * x^2 + {:.4f} * x + {:.4f}".format(A_quadratic, B_quadratic, C_quadratic))

# Побудова графіків
plt.scatter(x_data, y_data_linear, label='Лінійні дані', color='black')
plt.scatter(x_data, y_data_quadratic, label='Квадратичні дані', color='red')
plt.plot(x_vals, y_linear, label='Лінійна апроксимація', linestyle='dashed', color='black')
plt.plot(x_vals, y_quadratic, label='Квадратична апроксимація', linestyle='dashed', color='red')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Метод найменших квадратів')
plt.show()