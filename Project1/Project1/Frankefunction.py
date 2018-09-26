from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import itertools
from random import random, seed
from Methods import linregtools, MSE, R2, bootstrap

LAMBDA = 0.01
epsilon = 0.01

# Make data
n=20
degree = 20
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
xm, ym = np.meshgrid(x,y)

# FrankeFunction
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

f = FrankeFunction(xm, ym)
# f = f + np.random.randn(n,n)*0.1
object_f = linregtools(f,x,y,degree)

# Study numerical error
print("Polynomial fit of FrankeFunction in x, y of degree: ",degree)
X = object_f.X
XT = np.transpose(X)
XTX = np.matmul(XT,X)
inv_XTX = np.linalg.inv(XTX)
identity_approx = np.matmul(inv_XTX,XTX)
identity = np.identity(len(identity_approx))
print("|(X^TX)^-1(X^TX)-I| = ",np.linalg.norm(identity_approx-identity))
XTX[np.diag_indices_from(XTX)]+=LAMBDA
inv_XTX = np.linalg.inv(XTX)
identity_approx = np.matmul(inv_XTX,XTX)
identity = np.identity(len(identity_approx))
print("|(X^TX + I LAMBDA)^-1(X^TX + I LAMBDA)-I| = ",np.linalg.norm(identity_approx-identity), ", LAMBDA = ", LAMBDA)

# Plot regression of surface
freg = object_f.get_reg(LAMBDA)
fig = plt.figure()
ax = fig.gca(projection="3d")
surf = ax.plot_surface(xm, ym, freg, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(16))
ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

# Comparing R2 scores
freg =  object_f.get_reg()
print(" R2 score (ordinary least square): ", R2(f, freg))
freg =  object_f.get_reg(LAMBDA)
print(" R2 score (Ridge): ", R2(f, freg))
freg =  object_f.get_reg(LAMBDA,epsilon)
print(" R2 score (Lasso): ", R2(f, freg))

# Bootstrap comparison ordinary least square/Ridge/Lasso
training_data_sample_size=350
iterations = 10
bootstrap(training_data_sample_size, iterations, f, object_f)
bootstrap(training_data_sample_size, iterations, f, object_f, LAMBDA)
bootstrap(training_data_sample_size, iterations, f, object_f , LAMBDA, epsilon)

