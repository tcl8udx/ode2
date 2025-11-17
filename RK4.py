import numpy as np
import matplotlib.pyplot as plt

def RK1Solve(f, y0, nsteps, x0, xmax) -> np.array:
    h = (xmax - x0) / nsteps  # step size
    x = x0                     # independent variable
    y = y0                     # dependent variable to plot vs x
    points = [(x0, y0)]        # store points for plotting

    for i in range(nsteps - 1):
        k1 = h * f(x, y)
        y = y + k1
        x += h
        points.append((x, y))
    return np.array(points)

def RK2Solve(f, y0, nsteps, x0, xmax) -> np.array:
    h = (xmax - x0) / nsteps  # step size
    x = x0                     # independent variable
    y = y0                     # dependent variable to plot vs x
    points = [(x0, y0)]        # store points for plotting

    for i in range(nsteps - 1):
        k1 = h * f(x, y)
        k2 = h * f(x + h / 2, y + k1 / 2)
        y = y + k2
        x += h
        points.append((x, y))
    return np.array(points)

def RK4Solve(f, y0, nsteps, x0, xmax) -> np.array:
    h = (xmax - x0) / nsteps   # step size
    x = x0                     # independent variable
    y = y0                     # dependent variable to plot vs x
    points = [(x0, y0)]        # store points for plotting

    for i in range(nsteps - 1):
        k1 = h * f(x, y)
        k2 = h * f(x + h/2, y + k1/2)
        k3 = h * f(x + h/2, y + k2/2)
        k4 = h * f(x + h, y + k3)
        y = y + k1/6 + k2/3 + k3/3 + k4/6
        x += h
        points.append((x,y))
    return np.array(points)


# The differential equation to be solved
def fun1(x, y):
    return  x*y + x**5  # f = y'(x,y) = x * y(x) + x^5  
                    # solution: y(x) =  8*exp(x^2 / 2) - x^4 - 4x^2 - 8; with initial condition y(0)=0

# Solve our DEQ using RK1 or RK2 methods!
tg1 = RK1Solve(fun1, 0, 30, 0, 4)  # initial condition y(0)=0
tg2 = RK2Solve(fun1, 0, 30, 0, 4)
tg4 = RK4Solve(fun1, 0, 30, 0, 4)
x_exact = np.linspace(0, 4, 300)
y_exact = 8 * np.exp(x_exact**2 / 2) - x_exact**4 - 4*x_exact**2 - 8 # exact solution

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(tg1[:, 0], tg1[:, 1], 'r^', markersize=8, label='RK1 Solution')
plt.plot(tg2[:, 0], tg2[:, 1], 'g^', markersize=8, label='RK2 Solution')
plt.plot(tg4[:, 0], tg4[:, 1], 'b^', markersize=8, label='RK4 Solution')
plt.plot(x_exact, y_exact, 'k--', label='Exact Solution')

plt.title("RK4 demo")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.savefig("RK4.pdf")
# print("close plot window to exit")
# plt.show()
