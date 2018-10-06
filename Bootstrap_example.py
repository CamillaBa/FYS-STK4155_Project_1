import numpy as np
from random import random, seed
from Methods import linregtools, MSE, R2, bootstrap

# make data
n = 20
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
xm, ym = np.meshgrid(x,y)

# function to fit
def p(x,y):
    return x*x+0.5*x*y+y*y+x+y

f = p(xm, ym)
f = f + np.random.randn(n,n)*0.1
object = linregtools(f,x,y,3) # make third order polynomial fit to f

# boot strap (one iteration)
training_data_sample_size = 50
freg_bootstrap_step, training_data_id = object.get_bootstrap_step(training_data_sample_size)
for item in training_data_id:
    f[item[1],item[0]]=0.0
    freg_bootstrap_step[item[1],item[0]]=0.0
print("MSE: ", MSE(f,freg_bootstrap_step)," R2 score: ", R2(f, freg_bootstrap_step)) # check R2 and MSE on testing data

