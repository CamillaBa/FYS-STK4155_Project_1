from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from scipy import linalg
import itertools
from Methods import *
import time

# get the same random numbers each time we run the program
np.random.seed(10) 

# Make data
n=20; degree = 5; LAMBDA = 0.01; epsilon = 0.01 #some parameters
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
f = f + np.random.randn(n,n)*0.05
data_f = regdata(f,degree)
freg = data_f.get_reg()

#Nstart = -8; Nstop = 8; k = 12 #k=12 gives best fit on training data
#plot_R2_scores(data_f,Nstart,Nstop,"Fanke Function")
#plot_R2_scores_k_cross_validation(data_f,Nstart,Nstop,k,"Franke Function")

#degstart = 0; degend = 100; degstep = 5; 
#plot_R2_complexity(degstart,degend,degstep,f,"Franke Function")

#numerical_error(data_f,LAMBDA)

degstart = 5; degend = 100; degstep = 10; # warning, big degend results in long run time
plot_MSE_variance(degstart, degend, degstep, f)

## Get variance of betas
#var_covar_matrix = data_f.var_covar_matrix(freg)
#D = data_f.number_basis_elts
#var = np.zeros(D)
#for i in range(0,D):
#    var[i]=var_covar_matrix[i,i] # obtain the diagonal
#std = np.sqrt(var)

## find beta
#X = data_f.X
#z = data_f.z
#beta = data_f.get_beta(X,z)

## confidence intervals and associated basis elts
#confidence_interval, basis = [], []
#for i in range(0,D):
#    interval_i = [beta[i]-2*std[i],beta[i]+2*std[i]]
#    confidence_interval.append(interval_i)
#    powers_i = data_f.powers[i]
#    basis.append('$x^{}y^{}$'.format(powers_i[0],powers_i[1]))
    
## get the range of the confidence interval
#y_r = [beta[i] - confidence_interval[i][1] for i in range(0,D)]
#plt.figure()
#plt.bar(range(len(beta)), beta, yerr=y_r, alpha=0.2, align='center')
#plt.xticks(range(len(beta)), basis)
#plt.ylabel('Coefficients')
#plt.title(r'Confidence intervals of $\beta$')
#plt.show(block=False)

## write confidence intervals to file
#file = open("beta_confidence_intervals.txt","w+")
#for i in range(0,D):
#    file.write("{},{},{}\n".format(beta[i],
#                                 confidence_interval[i][0],
#                                 confidence_interval[i][1]))
#file.close()

## bootstrap comparison confidence interval
#sample_size = 200
#N = 1000000
#D = data_f.number_basis_elts
#betas = np.zeros((N,D))
#Ebeta, varbeta = np.zeros(D), np.zeros(D)
#for i in range(0,N):
#    betas[i,:] = data_f.bootstrap_step(sample_size)
#    Ebeta += betas[i,:]
#Ebeta = Ebeta/N
#for i in range(0,D):
#    varbeta[i]=np.sum((betas[:,i]-Ebeta[i])**2)/N
#std = np.sqrt(varbeta)

## confidence intervals and associated basis elts
#confidence_interval, basis = [], []
#for i in range(0,D):
#    interval_i = [Ebeta[i]-2*std[i],Ebeta[i]+2*std[i]]
#    confidence_interval.append(interval_i)
#    powers_i = data_f.powers[i]
#    basis.append('$x^{}y^{}$'.format(powers_i[0],powers_i[1]))

#y_r = [Ebeta[i] - confidence_interval[i][1] for i in range(0,D)]
#plt.figure()
#plt.bar(range(len(Ebeta)), Ebeta, yerr=y_r, alpha=0.2, align='center')
#plt.xticks(range(len(Ebeta)), basis)
#plt.ylabel('Coefficients')
#plt.title(r'Confidence intervals of $\beta$ (bootstrap, $N=${}, sample size={})'.format(N,sample_size))
#plt.show()

## write confidence intervals to file
#file = open("beta_confidence_intervals_bootstrap.txt","w+")
#for i in range(0,D):
#    file.write("{},{},{}\n".format(Ebeta[i],
#                                 confidence_interval[i][0],
#                                 confidence_interval[i][1]))
#file.close()

plt.show()