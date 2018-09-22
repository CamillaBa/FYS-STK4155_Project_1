import numpy as np
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
        if len(args) == 0:
            # get beta (Ordinary least square)
            beta = self.__get_beta(self.X,self.z)
        elif (len(args) == 1 and type(args[0])==float):
            # get beta (Ridge regression)
            LAMBDA = args[0]
            beta = self.__get_beta(self.X,self.z,LAMBDA)
        elif (len(args) == 2 and type(args[0])==float and type(args[1])==float):
            # get beta (Lasso regression)
            LAMBDA = args[0]
            epsilon = args[1]
            beta = self.__get_beta(self.X,self.z,LAMBDA,epsilon)
        reg = self.__regression(self.xm,self.ym,beta)
        return reg

    # get beta (given X and z)
    def __get_beta(self, X, z,*args):
        XT = np.transpose(X)
        beta = np.matmul(XT,X)

        #Ridge parameter LAMBDA
        if len(args) >= 1:
            LAMBDA = args[0] 
            beta[np.diag_indices_from(beta)]+=LAMBDA
        beta = np.linalg.inv(beta)    
        beta = np.matmul(beta,XT)
        beta = np.matmul(beta,z)

        #Shooting algorithm for Lasso
        if len(args)>1:
            epsilon = args[1]
            D = self.number_basis_elts
            ints = np.arange(0,D,1)
            beta_old = 0.0
            while np.sum(np.abs(beta-beta_old))>=epsilon:
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
    def get_bootstrap_step(self, samplesize,*args):
        mn =  self.mn
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
                znew[i]=self.z[counter]
                Xnew[i,:]=self.X[counter,:]
                training_data_id.append(self.correspondence[counter]) #retain pair (i,j) corresponding to counter
            if len(args) == 0:
                betanew = self.__get_beta(Xnew,znew)
            elif len(args) == 1:
                LAMBDA=args[0]
                betanew = self.__get_beta(Xnew,znew,LAMBDA)
            elif len(args) == 2:
                LAMBDA=args[0]
                epsilon = args[1]
                betanew = self.__get_beta(Xnew,znew,LAMBDA,epsilon)
            else:
                print("Wrong arguments.")
                return (0, training_data_id)
            regnew = self.__regression(self.xm,self.ym,betanew)
            return (regnew, training_data_id)



