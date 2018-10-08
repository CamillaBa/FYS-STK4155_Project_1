import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
import itertools
from Methods import *
from imageio import imread

# get the same random numbers each time we run the program
np.random.seed(10) 

degree = 10; LAMBDA = 0.00000001; epsilon = 0.001

# Loading the terrain as np.array and normalize it
terrain1 = np.array(imread('SRTM_data_Norway_1.tif'))
max_height = np.amax(terrain1)
terrain1 = terrain1/max_height

# Extracting small sample from 'SRTM_data_Norway_1.tif'
imin, jmin = 0, 0; 
imax, jmax = 50, 50;
x = np.linspace(0, 1, imax-imin)
y = np.linspace(0, 1, jmax-jmin)
xm, ym = np.meshgrid(x,y)
terrain1_sample = terrain1[imin:imax,jmin:jmax]

# initializing regression data

data = regdata(terrain1_sample,degree)

#===========================================================================================

degstart = 5; degend = 25; degstep = 5; 
plot_R2_complexity(degstart,degend,degstep,terrain1_sample,"terrain sample", LAMBDA = 0.00000001)

#===========================================================================================

#Nstart = -18; Nstop =3; k = 10
#plot_R2_scores(data,Nstart,Nstop,"terrain sample")
#ols_best, ridge_best, lasso_best = plot_R2_scores_k_cross_validation(data,Nstart,Nstop,k,"terrain sample")
#plot_3D(lasso_best, "Terrain sample model (Lasso, training model, $\lambda=10^{-11})$")

#===========================================================================================

#degstart = 5; degend = 65; degstep = 10; # warning, big degend results in long run time
#plot_MSE_variance(degstart, degend, degstep, f)

#============================================================================================

#plot_3D(terrain1_sample,"Terrain sample")
#model = data.get_reg(LAMBDA,epsilon)
#plot_3D(model, "Terrain sample model (Lasso $\lambda={}$, degree = {})".format(LAMBDA,degree))
#print(R2(model, terrain1_sample))

## Show the terrain 2D
#plt.figure()
#plt.title('Terrain over Norway 1 (segment)')
#plt.imshow(terrain1_zoom , cmap='gray')
#plt.xlabel('x')
#plt.ylabel('y')

plt.show()