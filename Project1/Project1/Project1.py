from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import itertools
from random import random, seed
from Methods import linregtools

fig = plt.figure()
ax = fig.gca(projection="3d")

# Make data
n=20
nn=n*n
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
f = f + np.random.randn(n,n)*0.25
object_f = linregtools(f,x,y,5)

# MSE
def MSE(x,y):
    n = np.size(x)
    return 1.0/n*np.sum((x-y)**2)

# R2 score
def R2(x_true,x_predict):
    n = np.size(x_true)
    x_avg = 1.0/n*np.sum(x_true)
    enumerator = np.sum ((x_true-x_predict)**2)
    denominator = np.sum((x_true-x_avg)**2)
    return 1 - enumerator/denominator

# Bootstrap
def bootstrap(training_data_sample_size, iterations, object, *args):
    if len(args)==0:
        print("Bootstrap: Ordinary least squares")
    elif len(args)==1:
        print("Bootstrap: Ridge regression")
    elif len(args)==2:
        print("Bootstrap: LASSO regression")
    for i in range(1,iterations+1):
        freg_bootstrap_step, training_data_id = object.get_bootstrap_step(training_data_sample_size,*args)
        fnew = np.copy(f)
        for item in training_data_id:
            fnew[item[1],item[0]]=0.0
            freg_bootstrap_step[item[1],item[0]]=0.0
        print("step: ", i, " MSE: ", MSE(fnew,freg_bootstrap_step)," R2 score: ", R2(fnew, freg_bootstrap_step))

training_data_sample_size=350
iterations = 10
fnew = np.zeros((n,n))
LAMBDA = 0.01
epsilon = 0.01

bootstrap(training_data_sample_size, iterations, object_f)
bootstrap(training_data_sample_size, iterations, object_f, LAMBDA)
bootstrap(training_data_sample_size, iterations, object_f , LAMBDA, epsilon)

# Plot regression of surface
freg =  object_f.get_reg()
print(" R2 score: ", R2(f, freg))
surf = ax.plot_surface(xm, ym, freg, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()













# Variance of beta ??
#sigma2 = sum((zreg-z)**2)/(n-5-1)
#var_covar_matrix_beta = invXTX*sigma2
#var = np.diag(var_covar_matrix_beta)
#sigma = np.sqrt(var)

# Confidence intervals
#confidence_intervals = []
#for i in range(0,21):
#    print([beta[i]-2*sigma[i],beta[i]+2*sigma[i]])