import numpy as np

# 定义函数 f(x) = x*sin(3x) - exp(x)
def f(x):
    return x * np.sin(3 * x) - np.exp(x)

# 定义函数 f'(x)
def f_prime(x):
    return np.sin(3 * x) + 3 * x * np.cos(3 * x) - np.exp(x)

# Newton-Raphson 方法
def newton_raphson(x0, tol=1e-6, max_iter=1000):
    x_values = []  # 用来存储每次迭代的 x 值
    x = x0  # 初始猜测
    for i in range(max_iter):
        x_values.append(x)
        fx = f(x)
        fpx = f_prime(x)
        
        if abs(fpx) < 1e-10:  # 避免除以接近零的情况
            print("Derivative is too small, stopping iteration.")
            break

        # 使用 Newton-Raphson 公式更新 x 值
        x_new = x - fx / fpx
        
        # 检查是否收敛
        if abs(x_new - x) < tol:
            x_values.append(x_new)
            break
        
        x = x_new
    
    return np.array(x_values)

# 执行牛顿法
A1 = newton_raphson(-1.6)  # 使用新的 Newton-Raphson 方法计算 A1
np.save('A1.npy', A1)  # 保存 A1 的结果

# 二分法求解
def bisection_method(a, b, tol=1e-6, max_iter=100):
    mid_values = []
    for i in range(max_iter):
        mid = (a + b) / 2
        mid_values.append(mid)
        if abs(f(mid)) < tol or (b - a) / 2 < tol:
            break
        if f(a) * f(mid) < 0:
            b = mid
        else:
            a = mid
    return np.array(mid_values), i + 1

# 执行二分法
A2, iterations_bisection = bisection_method(-0.7, -0.4)  # 赋值给A2
np.save('A2.npy', A2)  # 保存中点值

# 生成A3，包含牛顿法和二分法的迭代次数
A3 = np.array([len(A1) - 1, iterations_bisection])  # A1 的长度减1作为牛顿法的迭代次数
np.save('A3.npy', A3)

# 矩阵和向量定义
A = np.array([[1, 2], [-1, 1]])
B = np.array([[2, 0], [0, 2]])
C = np.array([[2, 0, -3], [0, 0, -1]]) 
D = np.array([[1, 2], [2, 3], [-1, 0]])
x = np.array([1, 0])  # 调整为一维数组
y = np.array([0, 1])  # 调整为一维数组
z = np.array([1, 2, -1])  # 调整为一维数组

# (a) A + B
A4 = A + B  # 赋值给A4
np.save('A4.npy', A4)

# (b) 3x - 4y
A5 = 3 * x - 4 * y  # 赋值给A5，x和y现在是一维数组
np.save('A5.npy', A5)

# (c) Ax
A6 = np.dot(A, x)  # 赋值给A6，结果调整为一维数组
np.save('A6.npy', A6)

# (d) B(x - y)
A7 = np.dot(B, (x - y))  # 赋值给A7，结果调整为一维数组
np.save('A7.npy', A7)

# (e) Dx
A8 = np.dot(D, x)  # 赋值给A8，结果调整为一维数组
np.save('A8.npy', A8)

# (f) Dy + z
A9 = np.dot(D, y) + z  # 赋值给A9，结果调整为一维数组
np.save('A9.npy', A9)

# (g) AB
A10 = np.dot(A, B)  # 赋值给A10
np.save('A10.npy', A10)

# (h) BC
A11 = np.dot(B, C)  # 赋值给A11
np.save('A11.npy', A11)

# (i) CD
A12 = np.dot(C, D)  # 赋值给A12
np.save('A12.npy', A12)
