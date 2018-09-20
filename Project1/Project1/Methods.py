import numpy as np
class ols:
    def __init__(self, f, x, y, order):
        self.x = x; self.y = y; self.order = order
        xm, ym = np.meshgrid(x,y)
        self.xm = xm; self.ym = ym
        m = len(f[0,:]); n = len(f); mn = m*n
        z = np.zeros(mn); xy = np.zeros((mn,2))

        self.correspondence = []

        # making a sequence xy containing the pairs (x_i,y_j) for i,j=0,...,n, and a sequence z with matching pairs z_ij = f(x_i, y_j)
        counter = 0
        for i in range(0,m):
            for j in range(0,n):
                z[counter]=f[j,i] #wtf???
                xy[counter,:] = [x[i],y[j]]
                self.correspondence.append([i,j]) #saves the 1-1 correspondence: {counter} <-> {(i,j)} for later
                counter+=1
        self.z = z

        # make X
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

        # get beta
        beta = self.__get_beta(X,z)
        self.reg = self.__regression(xm,ym,beta)

        zreg = np.zeros(mn)
        counter = 0
        for i in range(0,m):
            for j in range(0,n):
                zreg[counter]=self.reg[j,i]
                counter+=1
        self.zreg=zreg

    # returns regression from ordinary least square
    def get_reg_vector(self):
        return self.reg

    # returns true values as a sequence (for comparison purposes)
    def get_true_vector(self):
        return self.z

    # returns regression values as a sequence (for comparison purposes)
    def get_prediction(self):
        return self.zreg

    # get beta (given X and z)
    def __get_beta(self, X, z):
        XT = np.transpose(X)
        beta = np.matmul(XT,X)
        beta = np.linalg.inv(beta)
        beta = np.matmul(beta,XT)
        beta = np.matmul(beta,z)
        return beta

    # least squares regression
    def __regression(self,x,y,beta):
        order = self.order
        s=0
        counter = 0
        # loop that adds terms of the form beta*x^j*y^k such that j+k<=5
        for j in range(0,order + 1):
            k = 0
            while j+k <= order:
                s+= beta[counter]*(x**j)*(y**k)
                counter +=1
                k+=1
        return s

    # bootstrap step
    def get_bootstrap_step(self, samplesize):
        n = len(self.z)
        if not (1 <= samplesize and samplesize < n):
            print("Error. Choose samplesize between 1 and ", n,".")
            return 0
        else:
            samplesize = int(samplesize)
            integers = np.random.randint(low=0, high=n-1, size=samplesize)
            znew =  np.zeros(samplesize)
            Xnew = np.zeros((samplesize,self.number_basis_elts))
            training_data_id = []
            for i in range(0,samplesize):
                counter = integers[i]
                znew[i]=self.z[counter]
                Xnew[i,:]=self.X[counter,:]
                training_data_id.append(self.correspondence[counter]) #retain pair (i,j) corresponding to counter
            betanew = self.__get_beta(Xnew,znew)
            regnew = self.__regression(self.xm,self.ym,betanew)
            return (regnew, training_data_id)
        
