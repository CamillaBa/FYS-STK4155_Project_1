import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
import itertools
from Methods import *
import time
from imageio import imread

# Good settings for Ridge is degree = 150 and lambda = 0.000000001 
LAMBDA = 1e-13
epsilon = 0.001
degree = 15

# Load the terrain as np.array and normalize it
terrain1 = np.array(imread('SRTM_data_Norway_1.tif'))
max_height = np.amax(terrain1)
terrain1 = terrain1/max_height

# Convert terrain data to low quality
square_length = 50; square_size = int(square_length*square_length)
m = len(terrain1[0,:]); n = len(terrain1)
m_new = int(m/square_length); n_new = int(n/square_length)
terrain1_lq = np.zeros((n_new,m_new))
for i in range(0, n_new):
    for j in range(0, m_new):
        box = terrain1[i*square_length:(i+1)*square_length-1, j*square_length:(j+1)*square_length-1]
        terrain1_lq[i,j]=np.sum(box)/square_size
x = np.linspace(0,1,m_new)
y = np.linspace(0,1,n_new)
xm, ym = np.meshgrid(x,y)
data = regdata(terrain1_lq,degree)

#===================================================================================================

#degstart = 5; degend = 45; degstep = 10; 
#plot_R2_complexity(degstart,degend,degstep,terrain1_lq,"terrain sample", LAMBDA = 1e-13)

#===================================================================================================

#degstart = 5; degend = 45; degstep = 10; # warning, big degend results in long run time
#plot_MSE_variance(degstart, degend, degstep, terrain1_lq, LAMBDA = 1e-13)

#===================================================================================================

#Nstart = -15; Nstop =-10;
#plot_R2_scores(data,Nstart,Nstop,"terrain data (low quality)")

#===================================================================================================

plot_3D(terrain1_lq,"Terrain data (low quality)")
plot_3D(terrain1,"Terrain data")
model = data.get_reg(LAMBDA,epsilon)
plot_3D(model, "Terrain data (low quality) (Lasso, $\lambda=${}, degree = {})".format(LAMBDA,degree))
print(R2(model, terrain1_lq))

#===================================================================================================

#beta = data.get_beta(data.X,data.z,LAMBDA,epsilon)
#data = regdata(terrain1,degree)
#model = data.model(beta)
#plot_3D(model, "Terrain sample model (Lasso $\lambda={}$, degree = {})".format(LAMBDA,degree))
#print(R2(model,terrain1))

# Show the terrain
plt.figure()
plt.subplot(1, 2, 1)
plt.title('Terrain over Norway 1 (Low quality)')
plt.imshow(terrain1_lq , cmap='gray')
plt.xlabel('x')
plt.ylabel('y')
plt.subplot(1, 2, 2)
plt.title('Terrain over Norway 1')
plt.imshow(terrain1 , cmap='gray')
plt.xlabel('x')
plt.ylabel('y')

plt.show()