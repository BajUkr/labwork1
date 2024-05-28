import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

# Загрузка данных
data = [
    59, -35, 40, 59, 42, 26, -3, 36, -50, -2, -6, -59, 34, -48, -36, -21, 56, -52, -57, 57,
    34, 58, 34, -54, -31, 41, -40, -5, -61, 21, -19, -4, -8, -61, 57, -45, 40, -4, -59, 39,
    -27, 29, -21, 21, -8, -30, 36, -37, -54, -20, 44, 0, -34, -30, 44, 24, -26, 16, -16, 3,
    42, 35, -28, -43, -53, 41, 24, -39, -39, -2, -40, -55, -15, -42, 34, -12, -18, -21, 49,
    60, -17, -32, 50, -58, 17, 23, -25, -58, 12, 24, -31, -59, 31, -19, -31, 33, 64, -18, 5,
    -19, -20, 63, -27, 34, 17, -54, 15, -51, 29, 4, -26, 62, 33, -16, -14, 57, 15, 38, -37,
    -4, 39, 39, -2, -34, 2, 18, -23, -29, -50, 61, -44, 30, 59, -49, -53, 31, 23, -24, 33,
    40, 2, -17, 1, 24, -40, 58, 14, -14, 4, 6, -18, -4, -59, 13, -60, -49, 56, 39, 61, 60,
    -29, -48, 62, 8, 8, -61, -32, 56, 1, -45
]


print(len(data))
# Преобразуем данные в numpy массив
data = np.array(data)

# Разделим данные на 10 категорий для критерия χ²
num_bins = 10
hist, bin_edges = np.histogram(data, bins=num_bins)
expected = len(data) / num_bins

# Вычислим статистику χ²
chi_square_stat = ((hist - expected)**2 / expected).sum()

# Критическое значение для уровня значимости 0.05 и 9 степеней свободы
critical_value = stats.chi2.ppf(0.95, df=num_bins-1)

# Проверка гипотезы
chi_square_result = chi_square_stat < critical_value

# Критерий Колмогорова-Смирнова
# Сортируем данные
sorted_data = np.sort(data)

# Эмпирическая функция распределения
empirical_cdf = np.arange(1, len(data) + 1) / len(data)

# Теоретическая функция распределения
theoretical_cdf = stats.norm.cdf(sorted_data, np.mean(data), np.std(data))

# Статистика КС
D_plus = np.max(empirical_cdf - theoretical_cdf)
D_minus = np.max(theoretical_cdf - (np.arange(0, len(data)) / len(data)))
D_statistic = np.max([D_plus, D_minus])

# Критическое значение для уровня значимости 0.05
KS_critical_value = 1.36 / np.sqrt(len(data))

# Проверка гипотезы
ks_result = D_statistic < KS_critical_value

# Результаты
print(f"Chi-Square Statistic: {chi_square_stat}")
print(f"Chi-Square Critical Value: {critical_value}")
print(f"Chi-Square Test Result: {'Pass' if chi_square_result else 'Fail'}")
print(f"KS Statistic: {D_statistic}")
print(f"KS Critical Value: {KS_critical_value}")
print(f"KS Test Result: {'Pass' if ks_result else 'Fail'}")

# Визуализация
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(data, bins=num_bins, edgecolor='black')
plt.title('Histogram of Data')

plt.subplot(1, 2, 2)
plt.plot(sorted_data, empirical_cdf, label='Empirical CDF')
plt.plot(sorted_data, theoretical_cdf, label='Theoretical CDF')
plt.title('Empirical vs Theoretical CDF')
plt.legend()

plt.tight_layout()
plt.show()
