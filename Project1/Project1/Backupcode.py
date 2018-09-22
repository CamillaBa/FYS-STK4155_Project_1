#z = np.zeros(n*n)
#xy = np.zeros((n*n,2))

## Making a sequence xy containing the pairs (x_i,y_j) for i,j=0,...,n
#counter = 0
#for i in range(0,n):
#    for j in range(0,n):
#        z[counter]=f[j,i] #WTF???
#        xy[counter,:] = [x[i],y[j]]
#        counter+=1

## Make X
#X = np.zeros((nn,21))
#for i in range(0,nn):
#    counter = 0
#    for j in range(0,6):
#        k = 0
#        while j+k <= 5:
#            xi = xy[i,0]
#            yi = xy[i,1]
#            X[i,counter]= (xi**j)*(yi**k)
#            k+=1
#            counter+=1

## Make beta = (X^T X)^(-1)X^T
#XT = np.transpose(X)
#beta = np.matmul(XT,X)
#beta = np.linalg.inv(beta)
#invXTX = beta
#beta = np.matmul(beta,XT)
#beta = np.matmul(beta,z)

## Least squares regression
#def regression(x,y):
#    s=0
#    counter = 0
#    # Loop that adds terms of the form beta*x^j*y^k such that j+k<=5
#    for j in range(0,6):
#        k = 0
#        while j+k <= 5:
#            s+= beta[counter]*(x**j)*(y**k)
#            counter +=1
#            k+=1
#    return s
#freg = regression(xm,ym)

















    #    zreg = np.zeros(mn)
    #    counter = 0
    #    for i in range(0,m):
    #        for j in range(0,n):
    #            zreg[counter]=self.reg[j,i]
    #            counter+=1
    #    self.zreg=zreg
        
    ## returns true values as a sequence (for comparison purposes)
    #def get_true_vector(self):
    #    return self.z

    ## returns regression values as a sequence (for comparison purposes)
    #def get_prediction(self):
    #    return self.zreg

    #def __RSS(self,beta):
    #    mn = self.mn
    #    s=0
    #    for i in range(0,mn):
    #        zi = self.z[i]
    #        s+=(zi-np.dot(beta,X[i,:]))**2
    #    return s
    
    #def __LassoTerm(beta,LAMBDA):
    #    return LAMBDA*np.sum(np.absolute(beta))




        #    if len(args) != 3:
        #    XT = np.transpose(X)
        #    beta = np.matmul(XT,X)
        #    if len(args) == 1:
        #        LAMBDA = args[0]
        #        beta[np.diag_indices_from(beta)]+=LAMBDA
        #    beta = np.linalg.inv(beta)    
        #    beta = np.matmul(beta,XT)
        #    beta = np.matmul(beta,z)
        #else:
        #    LAMBDA = args[0]
        #    epsilon = args[1]
        #    N = args[2]
        #    def betaprobe():
        #        beta = np.random.uniform(low=-epsilon, high=epsilon, size=self.number_basis_elts)
        #        while sum(beta)>epsilon:
        #            beta = np.random.uniform(low=-epsilon, high=epsilon, size=self.number_basis_elts)
        #        stagnation = 0
        #        while stagnation < 20:
        #            dbeta = np.random.uniform(low=-epsilon, high=epsilon, size=self.number_basis_elts)*0.001
        #            betanew = beta + dbeta
        #            if  (self.__LassoCost(betanew,LAMBDA)<self.__LassoCost(beta,LAMBDA) and sum(betanew)<epsilon):
        #                beta = betanew
        #                stagnation = 0
        #            else:
        #                stagnation += 1
        #        return beta
        #    min = float("inf")
        #    attempts = 0
        #    while attempts < N:
        #        beta = betaprobe()
        #        mincandidate = self.__LassoCost(beta,LAMBDA)
        #        attempts+=1
        #        if mincandidate < min:
        #            min = mincandidate
        #            betacandidate = beta
        #    beta = betacandidate




                            #cj=0
                    #for i in range(0,mn):
                    #    ai = z[i]-np.dot(beta[no_j],X[i,no_j])
                    #    cj+=X[i,j]*ai
                    #cj*=2

    #def __LassoCost(self,beta,LAMBDA):
    #    X = self.X
    #    mn = self.mn
    #    s=0
    #    for i in range(0,mn):
    #        zi = self.z[i]
    #        s+=(zi-np.dot(beta,X[i,:]))**2
    #    s+=LAMBDA*np.sum(np.absolute(beta))
    #    return s


