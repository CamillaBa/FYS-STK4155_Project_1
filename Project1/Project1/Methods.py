import numpy as np
from scipy import linalg

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
    return 1.0 - enumerator/denominator

# Bootstrap
def bootstrap(training_data_sample_size, iterations, original_data, object, *args):
    f = original_data
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

class linregtools:
    def __init__(self, f, x, y, order):
        self.order = order
        xm, ym = np.meshgrid(x,y)
        self.xm = xm; self.ym = ym
        m = len(f[0,:]); n = len(f); mn = m*n; self.mn = mn
        z = np.zeros(mn); xy = np.zeros((mn,2))

        self.correspondence = []

        # Making a sequence xy containing the pairs (x_i,y_j) for i,j=0,...,n, and a sequence z with matching pairs z_ij = f(x_i, y_j)
        counter = 0
        for i in range(0,m):
            for j in range(0,n):
                z[counter]=f[j,i] #wtf???
                xy[counter,:] = [x[i],y[j]]
                self.correspondence.append([i,j]) #Saves the 1-1 correspondence: {counter} <-> {(i,j)} for later
                counter+=1
        self.z = z

        # Make X
        number_basis_elts=int((order+2)*(order+1)/2) #(order+1)th triangular number (number of basis elements for R[x,y] of degree <= order)
        X = np.zeros((mn,number_basis_elts))
        for i in range(0,mn):
            counter = 0
            for j in range(0,order+1):
                k = 0
                while j+k <= order:
                    xi = xy[i,0]
                    yi = xy[i,1]
                    X[i,counter]= (xi**j)*(yi**k)
                    k+=1
                    counter+=1
        self.X = X
        self.number_basis_elts = number_basis_elts

    # Regression
    def get_reg(self, *args):
        X=self.X
        z=self.z
        beta = self.__get_beta(X,z,*args)
        reg = self.__polynomial(beta)
        return reg

    # get beta (given X and z)
    def __get_beta(self, X, z,*args):
        XT = np.transpose(X)
        beta = np.matmul(XT,X)

        #Ridge parameter LAMBDA
        if len(args) >= 1:
            LAMBDA = args[0] 
            beta[np.diag_indices_from(beta)]+=LAMBDA
        beta = linalg.inv(beta)
        beta = np.matmul(beta,XT)
        beta = np.matmul(beta,z)

        #Shooting algorithm for Lasso
        if len(args)>1:
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

    # polynomial given beta
    def __polynomial(self,beta):
        xm = self.xm; ym = self.ym
        order = self.order
        s=0
        counter = 0
        # loop that adds terms of the form beta*x^j*y^k such that j+k<=5
        for j in range(0,order + 1):
            k = 0
            while j+k <= order:
                s+= beta[counter]*(xm**j)*(ym**k)
                counter +=1
                k+=1
        return s

    # bootstrap step
    def get_bootstrap_step(self, samplesize,*args):
        mn =  self.mn
        X = self.X
        z = self.z
        if not (1 <= samplesize and samplesize < mn):
            print("Error. Choose samplesize between 1 and ", mn-1,".")
            return 0
        else:
            samplesize = int(samplesize)
            integers = np.random.randint(low=0, high=mn-1, size=samplesize)
            znew =  np.zeros(samplesize)
            Xnew = np.zeros((samplesize,self.number_basis_elts))
            training_data_id = []
            for i in range(0,samplesize):
                counter = integers[i]
                znew[i]=z[counter]
                Xnew[i,:]=X[counter,:]
                training_data_id.append(self.correspondence[counter]) #retain pair (i,j) corresponding to counter
            betanew = self.__get_beta(Xnew,znew,*args)
            regnew = self.__polynomial(betanew)
            return (regnew, training_data_id)



