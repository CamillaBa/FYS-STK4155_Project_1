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
