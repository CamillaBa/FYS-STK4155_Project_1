from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import itertools
from random import random, seed
from Methods import *
from imageio import imread

#Good settings for Ridge is degree = 150 and lambda = 0.000000001 
LAMBDA = 0.000000001 
epsilon = 0.01
degree = 150

# Load the terrain as np.array and normalize it
terrain1 = np.array(imread('SRTM_data_Norway_1.tif'))
max_height = np.amax(terrain1)
terrain1 = terrain1/max_height

# Convert terrain data to low quality
square_length = 50
square_size = int(square_length*square_length)
m = len(terrain1[0,:])
n = len(terrain1)
m_new = int(m/square_length)
n_new = int(n/square_length)
terrain1_LQ = np.zeros((n_new,m_new))
for i in range(0, n_new):
    for j in range(0, m_new):
        box = terrain1[i*square_length:(i+1)*square_length-1, j*square_length:(j+1)*square_length-1]
        terrain1_LQ[i,j]=np.sum(box)/square_size
x = np.linspace(0,1,m_new)
y = np.linspace(0,1,n_new)
xm, ym = np.meshgrid(x,y)
object_terrain1_LQ = linregtools(terrain1_LQ,x,y,degree)

# Comparing R2 scores
#terrain1_LQ_reg =  object_terrain1_LQ.get_reg()
#print(" R2 score (ordinary least square): ", R2(terrain1_LQ_reg, terrain1_LQ))
terrain1_LQ_reg =  object_terrain1_LQ.get_reg(LAMBDA)
print(" R2 score (Ridge): ", R2(terrain1_LQ_reg, terrain1_LQ))
#terrain1_LQ_reg =  object_terrain1_LQ.get_reg(LAMBDA,epsilon)
#print(" R2 score (Lasso): ", R2(terrain1_LQ_reg, terrain1_LQ))

# Plot regression
terrain1_LQ_reg = object_terrain1_LQ.get_reg(LAMBDA)
fig = plt.figure()
ax = fig.gca(projection="3d")
surf = ax.plot_surface(xm, ym, terrain1_LQ_reg, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Customize the z axis.
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

# Show the terrain
plt.figure()
plt.subplot(1, 2, 1)
plt.title('Terrain over Norway 1 (Low quality)')
plt.imshow(terrain1_LQ , cmap='gray')
plt.xlabel('x')
plt.ylabel('y')
plt.subplot(1, 2, 2)
plt.title('Terrain over Norway 1')
plt.imshow(terrain1 , cmap='gray')
plt.xlabel('x')
plt.ylabel('y')
plt.show()