import numpy as np
from scipy.integrate import quad
from scipy.special import chebyt

# Задана функція f(x)
def f(x):
    return np.sqrt(x**2 + 2) * np.cos(2.14 * x)

# Параметри для нормалізації інтервалу [1, 3.5] до [-1, 1]
a = 1
b = 3.5
normalize = lambda x: (2 * x - a - b) / (b - a)

# Кількість коефіцієнтів
n = 4

# Функція для обчислення коефіцієнтів c_k
def calculate_coefficients(k):
    # Функція, яку інтегруємо
    integrand = lambda t: f((b - a) * t / 2 + (a + b) / 2) * chebyt(k)(t)

    # Інтеграл для обчислення коефіцієнта
    result, _ = quad(integrand, -1, 1)

    # Норма полінома Чебишева
    norm_squared = quad(lambda t: chebyt(k)(t)**2 / np.sqrt(1 - t**2), -1, 1)[0]

    # Обчислення коефіцієнта c_k
    coefficient = result / norm_squared

    return coefficient

# Обчислення коефіцієнтів для кожного полінома Чебишева
coefficients = [calculate_coefficients(k) for k in range(n)]

# Виведення результатів
print("Коефіцієнти середньоквадратичного наближення:", coefficients)
