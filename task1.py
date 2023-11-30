import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import chebyt

# Задана функція f(x)
def f(x):
    return np.sqrt(x**2 + 2) * np.cos(2.14 * x)

# Нормалізація інтервалу [1, 3.5] до [-1, 1]
a = 1
b = 3.5
normalize = lambda x: (2 * x - a - b) / (b - a)

# Кількість вузлів Чебишева
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

# Функція для обчислення значень функції та наближення в вузлах Чебишева
def evaluate_approximation():
    # Вузли Чебишева
    cheb_nodes = np.cos((2 * np.arange(1, n + 1) - 1) * np.pi / (2 * n))

    # Обчислення значень функції та наближення в вузлах
    true_values = [f((b - a) * t / 2 + (a + b) / 2) for t in cheb_nodes]
    approx_values = [sum(coefficients[k] * chebyt(k)(t) for k in range(n)) for t in cheb_nodes]

    return cheb_nodes, true_values, approx_values

# Обчислення коефіцієнтів для кожного полінома Чебишева
coefficients = [calculate_coefficients(k) for k in range(n)]

# Виведення результатів
print("Коефіцієнти середньоквадратичного наближення:", coefficients)

# Отримання результатів
cheb_nodes, true_values, approx_values = evaluate_approximation()

# Вивід у консоль вузлів, значень у функції, значень у апроксимованому графіку та похибок
print("\n Вузол\t\tФункція\t\tАпроксимація\tПохибка\n")
for node, true_val, approx_val in zip(cheb_nodes, true_values, approx_values):
   relative_error = ((true_val - approx_val) / true_val) * 100 if true_val != 0 else 0
   print(f"{node:.6f}\t{true_val:.6f}\t{approx_val:.6f}\t{relative_error:.6f}")


# Побудова графіків
x_vals = np.linspace(a, b, 100)
plt.plot(x_vals, f(x_vals), label='Функція f(x)')

# Точки для функції в вузлах Чебишева
plt.scatter((b - a) * cheb_nodes / 2 + (a + b) / 2, true_values, c='blue', marker='o', label='Функція f(x) в вузлах')

# Точки для апроксимованого графіку в тих же вузлах
plt.scatter((b - a) * cheb_nodes / 2 + (a + b) / 2, approx_values, c='red', marker='x', label='Точки апроксимованого графіку')

# Лінія для апроксимованого графіку
plt.plot(x_vals, [sum(coefficients[k] * chebyt(k)((2 * x - a - b) / (b - a)) for k in range(n)) for x in x_vals], 'r--', label='Апроксимований графік')

plt.legend()
plt.xlabel('x')
plt.ylabel('Значення')
plt.title('Порівняння функції та наближення в вузлах Чебишева')
plt.show()