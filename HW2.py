import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 定义空间范围和步长
L = 4
dx = 0.1
x_positive = np.arange(0, L + dx, dx)  # x从0到L
x_full = np.arange(-L, L + dx, dx)     # x从-L到L
num_points = len(x_full)

# 初始化存储特征函数和特征值的数组
A1 = np.zeros((num_points, 5))
A2 = np.zeros(5)

# 定义微分方程
def schrodinger(x, y, epsilon):
    # y[0] = φ, y[1] = φ'
    dydx = [y[1], (x**2 - epsilon) * y[0]]
    return dydx

# 定义射击法函数
def shooting(epsilon, parity):
    # 根据偶奇性设置初始条件
    if parity == 'even':
        y0 = [1.0, 0.0]  # φ(0) = 1, φ'(0) = 0
    else:
        y0 = [0.0, 1.0]  # φ(0) = 0, φ'(0) = 1

    sol = solve_ivp(schrodinger, [0, L], y0, args=(epsilon,), t_eval=x_positive, method='RK45')
    y = sol.y[0]
    return y[-1]  # 返回在x=L处的φ值

# 开始求解前五个特征值和特征函数
for n in range(1, 6):
    # 判断偶奇性
    if n % 2 == 1:
        parity = 'even'  # 奇数n对应偶函数
    else:
        parity = 'odd'   # 偶数n对应奇函数

    # 初始化能量猜测值
    epsilon_lower = 2 * n - 2  # 下界
    epsilon_upper = 2 * n      # 上界

    # 使用二分法寻找合适的ε_n
    tol = 1e-6
    max_iter = 100
    for _ in range(max_iter):
        epsilon = (epsilon_lower + epsilon_upper) / 2
        phi_L = shooting(epsilon, parity)

        if abs(phi_L) < tol:
            break  # 找到合适的ε_n
        elif phi_L * shooting(epsilon_lower, parity) < 0:
            epsilon_upper = epsilon
        else:
            epsilon_lower = epsilon

    # 存储特征值
    A2[n - 1] = epsilon

    # 计算完整的特征函数
    if parity == 'even':
        y0 = [1.0, 0.0]
    else:
        y0 = [0.0, 1.0]

    sol = solve_ivp(schrodinger, [0, L], y0, args=(epsilon,), t_eval=x_positive, method='RK45')
    phi_positive = sol.y[0]

    # 根据偶奇性扩展到负半轴
    if parity == 'even':
        phi_full = np.concatenate((phi_positive[::-1], phi_positive[1:]))
    else:
        phi_full = np.concatenate((-phi_positive[::-1], phi_positive[1:]))

    # 归一化
    norm = np.trapz(phi_full**2, x_full)
    phi_full_normalized = phi_full / np.sqrt(norm)

    # 存储特征函数的绝对值
    A1[:, n - 1] = np.abs(phi_full_normalized)

    # 绘制特征函数
    plt.plot(x_full, phi_full_normalized, label=f'n={n}')

# 设置绘图参数
plt.xlabel('x')
plt.ylabel('$\phi_n(x)$')
plt.title('one-dimensional harmonic trapping potential')
plt.legend()
plt.grid(True)
plt.show()

# 输出特征值
print("eigenvalue:")
print(A2)

# A1和A2已经在代码中存储



