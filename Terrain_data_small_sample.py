from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import itertools
from random import random, seed
from Methods import linregtools, MSE, R2, bootstrap
from imageio import imread

LAMBDA = 0.01
epsilon = 0.01

# Load the terrain as np.array and normalize it
terrain1 = np.array(imread('SRTM_data_Norway_1.tif'))
max_height = np.amax(terrain1)
terrain1 = terrain1/max_height

# Regression on small rectangle from 'SRTM_data_Norway_1.tif'
imin = 0; imax = 50; jmin = 0; jmax = 50
m_zoom = imax-imin
n_zoom = jmax-jmin
i_zoom = np.arange(imin,imax,1)
j_zoom = np.arange(jmin,jmax,1)
terrain1_zoom = np.zeros((m_zoom,n_zoom))
terrain1_zoom[:,:] = terrain1[imin:imax,jmin:jmax]
x = np.linspace(0,1,m_zoom)
y = np.linspace(0,1,n_zoom )
xm, ym = np.meshgrid(x,y)

object_terrain1_zoom = linregtools(terrain1_zoom,x,y,15)

# Comparing R2 scores
terrain1_zoom_reg =  object_terrain1_zoom.get_reg()
print(" R2 score (ordinary least square): ", R2(terrain1_zoom_reg, terrain1_zoom))
terrain1_zoom_reg =  object_terrain1_zoom.get_reg(LAMBDA)
print(" R2 score (Ridge): ", R2(terrain1_zoom_reg, terrain1_zoom))
terrain1_zoom_reg =  object_terrain1_zoom.get_reg(LAMBDA,epsilon)
print(" R2 score (Lasso): ", R2(terrain1_zoom_reg, terrain1_zoom))

# Plot regression
terrain1_zoom_reg = object_terrain1_zoom.get_reg(LAMBDA)
fig = plt.figure()
ax = fig.gca(projection="3d")
surf = ax.plot_surface(xm, ym, terrain1_zoom_reg, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Customize the z axis.
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))


# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

# Show the terrain
plt.figure()
plt.title('Terrain over Norway 1 (segment)')
plt.imshow(terrain1_zoom_reg , cmap='gray')
plt.xlabel('x')
plt.ylabel('y')
plt.show()