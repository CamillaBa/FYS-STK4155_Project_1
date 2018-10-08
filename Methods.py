from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import time

# Variance
def var(f_model):
    n = np.size(f_model)
    f_model_mean = np.sum(f_model)/n
    #f_model_mean = np.mean(f_model)
    return np.sum((f_model-f_model_mean)**2)/n

#================================================================================================================

# Bias
def bias(f_true,f_model):
    n = np.size(f_model)
    #f_model_mean = np.sum(f_model)/n
    f_model_mean = np.mean(f_model)
    return np.sum((f_true-f_model_mean)**2)/n

#================================================================================================================

# MSE
def MSE(f_true,f_model):
    n = np.size(f_model)
    return np.sum((f_true-f_model)**2)/n

#================================================================================================================

# Extra term
def extra_term(f_true,f_model):
    n = np.size(f_model)
    f_model_mean = np.mean(f_model)
    return 2.0/n*np.sum((f_model_mean-f_true)*(f_model-f_model_mean))

#================================================================================================================

# SVD invert
def SVDinv(A):
    ''' Takes as input a numpy matrix A and returns inv(A) based on singular value decomposition (SVD).
    SVD is numerically more stable (at least in our case) than the inversion algorithms provided by
    numpy and scipy.linalg at the cost of being slower.
    '''
    U, s, VT = linalg.svd(A)
    D = np.zeros((len(U),len(VT)))
    for i in range(0,len(VT)):
        D[i,i]=s[i]
    UT = np.transpose(U); V = np.transpose(VT); invD = np.linalg.inv(D)
    return np.matmul(V,np.matmul(invD,UT))

#================================================================================================================

# R2 score
def R2(x_true,x_predict):
    n = np.size(x_true)
    x_avg = np.sum(x_true)/n
    enumerator = np.sum ((x_true-x_predict)**2)
    denominator = np.sum((x_true-x_avg)**2)
    return 1.0 - enumerator/denominator

#================================================================================================================

# get sub-entries of matrix A
def get_subset(A,indices):
    '''given an indexing set "indices", return the vector consisting of 
    entries A[i,j] where (i,j) is an entry in indices.'''
    N = len(indices)
    B = np.zeros(N)
    for k in range(0,N):
        i = indices[k][0]
        j = indices[k][1]
        B[k] = A[j,i]
    return B

#================================================================================================================

# k fold cross validation
def k_cross_validation(data,partition,*args):
    '''function that finds the best model from k-cross validation resampling.
    Given a partition, we single out the training data with the highest R2 score
    and return the the model and the indices that consitute the training and
    testing data.
    '''
    f = data.f; X = data.X; z = data.z; correspondence = data.correspondence
    bestR2score = float("-inf")
    k = len(partition)
    for i, test_data in enumerate(partition):
        training_data = [x for j,x in enumerate(partition) if j!=i]
        training_data = sum(training_data, [])
        beta = data.get_beta(X[training_data],z[training_data],*args)
        freg = data.model(beta)
        test_data = [correspondence[j] for j in test_data]
        training_data = [correspondence[j] for j in training_data]
        R2score = R2(get_subset(f,test_data), get_subset(freg,test_data))
        if R2score > bestR2score:
            bestR2score = R2score
            best = i
            bestmodel = np.copy(freg); best_test_data = np.copy(test_data); best_training_data = np.copy(training_data)
    return bestmodel, best_test_data, best_training_data 


#================================================================================================================

class regdata:
    def __init__(self, f, degree):
        # initializing variables
        m = len(f[0,:]); n = len(f);  mn = m*n; 
        x = np.linspace(0, 1, m); y = np.linspace(0, 1, n); z = np.zeros(mn); xy = np.zeros((mn,2)); 

        # initializing some self variables
        self.f = f; self.degree = degree; self.xm, self.ym = np.meshgrid(x,y); self.n=n;self.m=m;  self.mn = mn; self.correspondence = []
        
        # Making a sequence xy containing the pairs (x_i,y_j) for i,j=0,...,n, and a sequence z with matching pairs z_ij = f(x_i, y_j)
        counter = 0
        for i in range(0,m):
            for j in range(0,n):
                z[counter]=f[j,i] #wtf
                xy[counter,:] = [x[i],y[j]]
                self.correspondence.append([i,j]) #Saves the 1-1 correspondence: {counter} <-> {(i,j)} for later
                counter+=1
        self.z = z

        # Make X
        number_basis_elts=int((degree+2)*(degree+1)/2) #(degree+1)th triangular number (number of basis elements for R[x,y] of degree <= degree)
        X = np.zeros((mn,number_basis_elts))
        powers = []
        for i in range(0,mn):
            counter = 0
            for j in range(0,degree+1):
                k = 0
                while j+k <= degree:
                    xi = xy[i,0]
                    yi = xy[i,1]
                    X[i,counter]= (xi**j)*(yi**k)
                    powers.append([j , k])
                    k+=1
                    counter+=1
        self.X = X
        self.powers = powers
        self.number_basis_elts = number_basis_elts
        self.invXTX = linalg.inv(np.matmul(np.transpose(X),X))

    # Regression
    def get_reg(self, *args):
        '''Returns the polynomial fit as a numpy array. If *args is empty the fit is based on an ordinary least square.
        If *args contains a number LAMBDA, then the fit is found using Ridge for the given bias LAMBDA. If *args contains
        two numbers LAMBDA and epsilon, then the fit is found using lasso. See the function " __get_beta" for more details.'''

        X=self.X; z=self.z #relabeling self variables
        beta = self.get_beta(X,z,*args) #obtaining beta
        reg = self.model(beta) #obtaining model from coefficients beta
        return reg

    # Get beta (given X and z)
    def get_beta(self, X, z,*args):
        '''Returns coefficients for a given beta as a numpy array, found using either ordinary least square,
        Ridge or Lasso regression depending on the arguments. If *args is empty, then beta is found using
        ordinary least square. If *args contains a number it will be treated as a bias LAMBDA for a Ridge regression.
        If *args contains two numbers, then the first will count as a LAMBDA and the second as a tolerance epsilon.
        In this case beta is found using a shooting algorithm that runs until it converges up to the set tolerance.
        '''

        XT = np.transpose(X)
        beta = np.matmul(XT,X)                   
        if len(args) >= 1: #Ridge parameter LAMBDA
            LAMBDA = args[0] 
            beta[np.diag_indices_from(beta)]+=LAMBDA
        beta = SVDinv(beta)
        beta = np.matmul(beta,XT)
        beta = np.matmul(beta,z)

        #Shooting algorithm for Lasso
        if len(args)>=2:
            epsilon = args[1]
            D = self.number_basis_elts
            ints = np.arange(0,D,1)
            beta_old = 0.0
            while np.linalg.norm(beta-beta_old)>=epsilon:
                beta_old = np.copy(beta)
                for j in range(0,D):
                    aj = 2*np.sum(X[:,j]**2)
                    no_j = ints[np.arange(D)!=j]
                    cj = 2*np.sum(np.multiply(X[:,j],(z-np.matmul(X[:,no_j],beta[no_j]))))
                    if cj<-LAMBDA:
                        beta[j]=(cj+LAMBDA)/aj
                    elif cj > LAMBDA:
                        beta[j]=(cj-LAMBDA)/aj
                    else:
                        beta[j]=0.0
        return beta

    # Get model given beta
    def model(self,beta):
        '''Returns heigh values based on the coefficients beta as a matrix
        that matches the grid xm, ym. The degree of the polynomial equals self.degree.
        '''
        xm = self.xm; ym = self.ym; degree = self.degree #relabeling self variables
        s=0
        counter = 0
        # loop that adds terms of the form beta*x^j*y^k such that j+k<=5
        for j in range(0,degree + 1):
            k = 0
            while j+k <= degree: 
                s+= beta[counter]*(xm**j)*(ym**k)
                counter +=1
                k+=1
        return s

    def get_data_partition(self,k):
        ''' Creates a random partition of k (almost) equally sized parts of the array
        {1,2,...,mn}. This can be used to make training/testing data.
        '''
        mn =  self.mn; correspondence = self.correspondence
        indices = np.arange(mn)
        indices_shuffle = np.arange(mn)
        np.random.shuffle(indices_shuffle)
        partition = []
        for step in range(0,k):
            part = list(indices_shuffle[step:mn:k])
            #part = [correspondence[i] for i in part]
            partition.append(part) 
        return partition

    def bootstrap_step(self, samplesize, *args):
        '''Finds and returns the coefficient that determines a model (ols, Ridge or Lasso),
        depending on args*.
        '''
        mn =  self.mn; X = self.X; z = self.z;  #relabeling self variables
        integers = np.random.randint(low=0, high=mn-1, size=samplesize)
        znew =  z[integers]
        Xnew = X[integers,:]
        betanew = self.get_beta(Xnew,znew,*args)
        return betanew

    # Variance/ covariance matrix
    def var_covar_matrix(self,reg):
        ''' Returns the variance/covariance matrix for beta based on the given data.
        This matrix is derived from a statistical viewpoint, where one assumes beta to
        have a normal distribution.
        '''
        p = self.number_basis_elts; invXTX = self.invXTX; N = self.mn; f = self.f # Relabeling self variables
        sigma2=1.0/(N-p-1)*np.sum((f-reg)*(f-reg))
        return sigma2*invXTX # OBS! Based on matrix inversion. Inaccurate for  N,p>>0.

#================================================================================================================

def plot_3D(f,plottitle):
    ''' Simple function to create 3d plot of the given data f,
    with plotitle.
    '''

    m = len(f[0,:]); n = len(f);
    x = np.linspace(0, 1, m)
    y = np.linspace(0, 1, n);
    xm, ym = np.meshgrid(x,y)

    # Plot f
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    surf = ax.plot_surface(xm, ym, f, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
    ax.text2D(0.05, 0.95, plottitle, transform=ax.transAxes)
    ax.view_init(30, 60)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

#================================================================================================================

def numerical_error(data,LAMBDA):
    '''Rough numerical analysis of matrix inversions for this problem. Comparison of error and time usage
    of SVD (singular values decomposition) for matrix inversion against scipy.linalg inversion algorithm.
    Printing results to terminal.
    '''
    return_items = []

    degree = data.degree; m = data.m; n = data.n
    # Study numerical error and time for SVD
    print("Polynomial fit of FrankeFunction in x, y of degree ", degree," with grid size ", (m,n)," analysis:")
    print("")
    X = data.X; XT = np.transpose(X); XTX = np.matmul(XT,X) #Obtaining XTX
    start_time = time.time() # start meassuring time
    inv_XTX = linalg.inv(XTX) # inversion using scipi.linalg
    end_time = time.time()
    print("Inverting XTX without SVD", "--- %s seconds ---" % (end_time - start_time)); return_items.append(end_time - start_time)
    inv_XTX_ = np.copy(inv_XTX) # storing inversion of XTX for later
    start_time = time.time()
    inv_XTX = SVDinv(XTX)
    end_time = time.time()
    print("Inverting XTX with SVD", "--- %s seconds ---" % (end_time - start_time)); return_items.append(end_time - start_time)
    print(' ')
    I_approx_ = np.matmul(inv_XTX_,XTX); # approximate I (no SVD)
    I = np.identity(len(I_approx_)); # obtaining analytical I
    output = np.linalg.norm(I_approx_-I)
    print("|(X^TX)^-1(X^TX)-I| = ",output, " (no SVD)"); return_items.append(output)
    I_approx = np.matmul(inv_XTX,XTX) # approximate I (SVD)
    output = np.linalg.norm(I_approx-I)
    print("|(X^TX)^-1(X^TX)-I| = ",np.linalg.norm(I_approx-I), " (SVD)"); return_items.append(output)
    XTX[np.diag_indices_from(XTX)]+=LAMBDA
    inv_XTX = linalg.inv(XTX)
    I_approx_ = np.matmul(inv_XTX,XTX) # approximate I (no SVD)
    output = np.linalg.norm(I_approx_-I)
    print("|(X^TX + I LAMBDA)^-1(X^TX + I LAMBDA)-I| = ",output , ", LAMBDA = ", LAMBDA, " (no SVD)"); return_items.append(output)
    inv_XTX = SVDinv(XTX)
    I_approx = np.matmul(inv_XTX,XTX)
    output = np.linalg.norm(I_approx-I)
    print("|(X^TX + I LAMBDA)^-1(X^TX + I LAMBDA)-I| = ",output, ", LAMBDA = ", LAMBDA, " (SVD)"); return_items.append(output)
    print(' ')
    
    return return_items

#================================================================================================================

def plot_R2_scores(data,Nstart,Nstop,name):
    ''' This function makes a plot of the R2 scores vs Lambda of the different regression methods,
    for a given dataset.'''

    degree = data.degree; f = data.f # obtaining class data
    N = Nstop-Nstart # number of lambdas
    epsilon = 0.01
    lambdas = np.zeros(N)
    R2_ols = np.zeros(N)
    R2_Ridge = np.zeros(N)
    R2_Lasso = np.zeros(N)
    for i in range(0,N):
        LAMBDA = 10**(Nstart+i)
        lambdas[i]=LAMBDA
        R2_ols[i]=R2(f, data.get_reg())
        R2_Ridge[i]=R2(f, data.get_reg(LAMBDA))
        R2_Lasso[i]=R2(f, data.get_reg(LAMBDA,epsilon))
        print("Completed lambda: ", LAMBDA, " Completion: {:.1%}".format(float(i)/(N-1)))
    plotitle = '$R^2$ score of degree {} polynomial fit on {}'.format(degree,name)
    plt.figure()
    plt.plot(np.log10(lambdas),R2_ols)
    plt.plot(np.log10(lambdas),R2_Ridge)
    plt.plot(np.log10(lambdas),R2_Lasso,'--')
    plt.axis([Nstart, N+Nstart-1, 0, 1])
    plt.xlabel('log $\lambda$')
    plt.ylabel('$R^2$ score')
    plt.legend(('Ordinary least square','Ridge','Lasso'))
    plt.title(plotitle)
    plt.grid(True)
    plt.show(block=False)

#================================================================================================================

def plot_R2_scores_k_cross_validation(data,Nstart,Nstop,k,name):
    ''' This function makes a plot of the R2 scores vs LAMBDA of the best iteration from a k-fold cross validation on 
    the data set from the given data. Best in the sense that the fit had the highest R2 score on testing data. The same 
    partition of the data set is used for each lambda, and each time we select the best training data on which we base the model.
    See "k_cross_validation" for more details.'''

    degree = data.degree; f = data.f # obtaining class data
    N = Nstop-Nstart # number of lambdas
    epsilon = 0.001

    # Comparing R2 scores, regression with fixed degree, variable LAMBDA
    lambdas = np.zeros(N)
    partition = data.get_data_partition(k)
    R2_ols_test_data = np.zeros(N)
    R2_ols_training_data = np.zeros(N)
    R2_Lasso_test_data = np.zeros(N)
    R2_Lasso_training_data = np.zeros(N)
    R2_Ridge_test_data = np.zeros(N)
    R2_Ridge_training_data = np.zeros(N)

    # Best OLS model on training data from k-fold cross validation
    freg, test_data_id, training_data_id = k_cross_validation(data,partition) 
    R2_ols_test_data[:] = R2(get_subset(f,test_data_id),get_subset(freg,test_data_id))
    R2_ols_training_data[:] = R2(get_subset(f,training_data_id),get_subset(freg,training_data_id))

    ols_best = np.copy(freg)
    ridge_best_list, lasso_best_list = [],[]
    for i in range(0,N): 
        LAMBDA = 10**(Nstart+i)
        lambdas[i]=LAMBDA

        # Best Ridge model on training data from k-fold cross validation
        freg, test_data_id, training_data_id = k_cross_validation(data,partition,LAMBDA)
        ridge_best_list.append(freg)
        R2_Ridge_test_data[i] = R2(get_subset(f,test_data_id),get_subset(freg,test_data_id))
        R2_Ridge_training_data[i] = R2(get_subset(f,training_data_id),get_subset(freg,training_data_id))

        # Best Lasso model on training data from k-fold cross validation
        freg, test_data_id, training_data_id = k_cross_validation(data,partition,LAMBDA,epsilon)
        lasso_best_list.append(freg)
        R2_Lasso_test_data[i] = R2(get_subset(f,test_data_id),get_subset(freg,test_data_id))
        R2_Lasso_training_data[i] = R2(get_subset(f,training_data_id),get_subset(freg,training_data_id))

        print("Completed lambda: ", LAMBDA, " Completion: {:.1%}".format(float(i)/(N-1)))

    ibest_ridge = np.argmax(R2_Ridge_test_data)
    ibest_lasso = np.argmax(R2_Lasso_test_data)
    ridge_best = ridge_best_list[ibest_ridge]
    lasso_best = lasso_best_list[ibest_lasso]

    plotitle = '$R^2$ scores of degree {} polynomial fit on {}, $k=${}'.format(degree,name,k)
    plt.figure()
    plt.plot(np.log10(lambdas),R2_ols_test_data)
    plt.plot(np.log10(lambdas),R2_ols_training_data,'--')
    plt.plot(np.log10(lambdas),R2_Ridge_test_data)
    plt.plot(np.log10(lambdas),R2_Ridge_training_data,'--')
    plt.plot(np.log10(lambdas),R2_Lasso_test_data)
    plt.plot(np.log10(lambdas),R2_Lasso_training_data,'--')
    plt.axis([Nstart, Nstart+N-2, 0, 1])
    plt.xlabel('log $\lambda$')
    plt.ylabel('$R^2$ score')
    if (np.amax(R2_ols_test_data)> 0 and np.amax(R2_ols_training_data)> 0):
        plt.legend(('OLS: test data', 'OLS: training data','Ridge: test data', 'Ridge: training data','Lasso: test data', 'Lasso: training data'))
    elif (np.amax(R2_ols_test_data)<= 0 and np.amax(R2_ols_training_data)> 0):
        plt.legend(('OLS: test data (negative)', 'OLS: training data','Ridge: test data', 'Ridge: training data','Lasso: test data', 'Lasso: training data'))
    elif (np.amax(R2_ols_test_data)> 0 and np.amax(R2_ols_training_data)<= 0):
        plt.legend(('OLS: test data', 'OLS: training data (negative)','Ridge: test data', 'Ridge: training data','Lasso: test data', 'Lasso: training data'))
    elif (np.amax(R2_ols_test_data)<= 0 and np.amax(R2_ols_training_data)<= 0):
        plt.legend(('OLS: test data (negative)', 'OLS: training data (negative)','Ridge: test data', 'Ridge: training data','Lasso: test data', 'Lasso: training data'))
    plt.title(plotitle)
    plt.grid(True)
    plt.show(block=False)

    return ols_best, ridge_best, lasso_best

#================================================================================================================

def plot_R2_complexity(degstart,degend,degstep,f,name, LAMBDA = 0.00001, epsilon = 0.001):
    ''' Comparing R2 scores, regression with fixed LAMBDA, variable degree as well as variance and Bias
    Plotting the result.
    '''
    degrees = np.arange(degstart,degend+1,degstep)
    N = len(degrees)
    R2_ols, R2_Ridge, R2_Lasso = np.zeros(N), np.zeros(N), np.zeros(N)
    for i, degree in enumerate(degrees):
        data_f = regdata(f,degree)
        R2_ols[i]=R2(f, data_f.get_reg())
        R2_Ridge[i]=R2(f, data_f.get_reg(LAMBDA))
        R2_Lasso[i]=R2(f, data_f.get_reg(LAMBDA,epsilon))
        print("Completed degree: ", degree, " Completion: {:.1%}".format(float(i)/(N-1)))
    plotitle = '$R^2$ score of polynomial fit on {} with $\lambda=${}'.format(name,LAMBDA)
    plt.figure()
    plt.plot(degrees,R2_ols)
    plt.plot(degrees,R2_Ridge)
    plt.plot(degrees,R2_Lasso,'--')
    plt.xlabel('degree of fitting polynomial')
    plt.ylabel('$R^2$ score')
    plt.axis([degstart,degend, 0, 1])
    plt.legend(('Ordinary least square','Ridge','Lasso'))
    plt.title(plotitle)
    plt.grid(True)
    plt.show(block=False)

#================================================================================================================

def plot_MSE_variance(degstart, degend, degstep, f, LAMBDA = 0.01, epsilon = 0.001):
    # Comparing MSE, bias, variance and additional terms as function of complexity.
    degrees = np.arange(degstart,degend+1,degstep)
    N = len(degrees)
    data = regdata(f,5)
    fvar = np.zeros(N); fbias = np.zeros(N); fMSE = np.zeros(N); fextra_terms = np.zeros(N)

    # function for plotting
    def makeplot(methodname, *args, k=0):
        print(methodname)
        for i, degree in enumerate(degrees):
            data = regdata(f,degree)
            if k == 0:
                freg = data.get_reg(*args)
                fvar[i], fbias[i], fMSE[i], fextra_terms[i] =  var(freg), bias(f,freg), MSE(f,freg), extra_term(f,freg)
            elif k>0:
                freg, test_data_id, training_data_id = k_cross_validation(data, partition, *args)
                ft = get_subset(f,test_data_id)
                fregt = get_subset(freg,test_data_id)
                fvar[i], fbias[i], fMSE[i], fextra_terms[i] =  var(fregt), bias(ft,fregt), MSE(ft,fregt), extra_term(ft,fregt)
            print("Completed degree: ", degree, " Completion: {:.1%}".format(float(degree-degstart)/(degend-degstart)))
        plt.figure()
        plt.plot(degrees, fvar)
        plt.plot(degrees, fbias)
        plt.plot(degrees, fMSE,'--')
        plt.plot(degrees, fextra_terms)
        plt.xlabel('degree')
        plt.ylabel('Variance, bias, and MSE')
        plt.legend(('Variance','Bias','MSE','Additional term'))
        plt.grid(True)
        plt.show(block=False)

    #It is a good idea to comment out the plots that you dont need


    # Ordinary least square plot
    makeplot("Ordinary least squares")
    plt.title("Error of  ordinary least squares")

    # Ridge plot
    makeplot("Ridge regression",LAMBDA)
    plt.title("Error of Ridge regression, $\lambda=${}".format(LAMBDA))

    # Lasso plot
    makeplot("Lasso regression",LAMBDA,epsilon)
    plt.title("error of lasso regression, $\lambda=${}".format(LAMBDA))

    # k-cross validation
    partition = data.get_data_partition(12)

    # Ordinary least square plot
    makeplot("Ordinary least squares 12-cross validation", k=12)
    plt.title("Error of ordinary least squares \n using 12-cross validation")
    
    # Ridge plot
    makeplot("Ridge regression 12-cross validation", LAMBDA, k=12)
    plt.title("Error of Ridge regression, $\lambda=${} \n using 12-cross validation".format(LAMBDA))

    # Lasso plot
    makeplot("Lasso regression 12-cross validation", LAMBDA, epsilon, k=12)
    plt.title("Error of Lasso regression, $\lambda=${} \n using 12-cross validation".format(LAMBDA))





