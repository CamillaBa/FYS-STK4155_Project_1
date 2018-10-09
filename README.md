# FYS-STK4155_Project_1

Contains 4 python files:

1) Methods.py 
2) Frankefunction.py
3) Terrain_data_low_quality.py
4) Terrain_data_small_sample.py


File 1) consists of all the regression methods that we developed
during this project. It contains stardard formulas implemented as functions,
such as variance, MSE, R2 score, etc, but also more tailored functions
that makes plots of R2 scores of training and test data and the like.

Most important is the class "regdata" which stores data needed to perform
linear regression on a data set. The functions of this class perform the variuos
regression methods on the given data. This file is also discussed in the methods section of our report.

Another class is "k_cross_validation". It takes as input an object "data" of type regdata,
and a partition of the data into k (evenly sized) subsets. It has class functions that performs k-cross validation
for the given subsets and calcuates R2 scores, MSE, bias, and variance.
It stores these values as self variables for usage in plots.

File 2) is a file that runs different methods from Methods.py 
to study Franke function.  Should you run it as is, it will print

"Polynomial fit of FrankeFunction in x, y of degree  50  with grid size  (20, 20)  analysis:

Inverting XTX without SVD --- 0.036899566650390625 seconds ---
Inverting XTX with SVD --- 0.32114124298095703 seconds ---

|(X^TX)^-1(X^TX)-I| =  9225.613585292722  (no SVD)
|(X^TX)^-1(X^TX)-I| =  36.557123413673416  (SVD)
|(X^TX + I LAMBDA)^-1(X^TX + I LAMBDA)-I| =  0.0001001993974806641 , LAMBDA =  1e-08  (no SVD)
|(X^TX + I LAMBDA)^-1(X^TX + I LAMBDA)-I| =  0.00047783218944391116 , LAMBDA =  1e-08  (SVD)"

for different degrees and plots that compares singular value decomposition to
standard inversion tools. Also, you will finds plots giving the estimates of the
confidence interval of the coefficients beta, using a formula, or using bootstrapping.

Some of the file contents is commented out because it takes a while to run. We used
these parts to create plots for the report.

File 3) is a file that is used to study a low quality version of our terrain data using
methods from methods.py. If you run it, it will plot four plots, three 3D plots
that compare the high resolution to the low resolution, including a degree 15 model using Lasso.
The last plot is a 2D comparison plot. Again, some of the file contents is commented out.

File 4) is a file that is used to study the small terrain sample. If you run it as is,
it plots a 3D version of the true data. In addition, it plots R2 scores as functions of Lambda and
complexity for some models.

Some functionality is commented out.

It prints:

"
Completed degree:  5  Completion: 0.0%
Completed degree:  7  Completion: 16.7%
Completed degree:  9  Completion: 33.3%
Completed degree:  11  Completion: 50.0%
Completed degree:  13  Completion: 66.7%
Completed degree:  15  Completion: 83.3%
Completed degree:  17  Completion: 100.0%
Completed lambda:  1e-08  Completion: 0.0%
Completed lambda:  1e-07  Completion: 20.0%
Completed lambda:  1e-06  Completion: 40.0%
Completed lambda:  1e-05  Completion: 60.0%
Completed lambda:  0.0001  Completion: 80.0%
Completed lambda:  0.001  Completion: 100.0%
Completed lambda:  1e-08  Completion: 0.0%
Completed lambda:  1e-07  Completion: 20.0%
Completed lambda:  1e-06  Completion: 40.0%
Completed lambda:  1e-05  Completion: 60.0%
Completed lambda:  0.0001  Completion: 80.0%
Completed lambda:  0.001  Completion: 100.0%
"
