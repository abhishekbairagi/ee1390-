import numpy as np 
import matplotlib.pyplot as plt 

def dir_vec(AB):
	return np.matmul(AB,dvec)
def norm_vec(AB):
	return np.matmul(omat,np.matmul(AB,dvec))


dvec = np.array([-1,1])
omat = np.array([[0,1] , [-1,0]])

def line_intersect(n1 ,n2 ,p):
	N = np.vstack((n1,n2))
	return np.matmul(np.linalg.inv(N),p)

n1 = np.array([2, 1])
n2 = np.array([1, -1])
p  = np.array([3, 1])

C = line_intersect(n1, n2 , p)
print "The point of intersection is ",C 

D = np.array([1,-1])

CD = np.vstack((C,D)).T

n= norm_vec(CD)
d= dir_vec(CD)

slope_tangent = n
normmal_tangent = d

print 'slope_tangent' , slope_tangent[1]/slope_tangent[0]
#print 'normmal_tangent' , normmal_tangent[1]/normmal_tangent[0]

print 'The equation of tangent is', normmal_tangent,"X = ", n2


P =np.array([0 , -0.75])

len = 100000
lam_1 = np.linspace(-10,10,len)

l2a = np.array([1,0])
l2b = np.array([0,-1])
l1a = np.array([1.5,0])
l1b = np.array([0,3])



x_DP = np.zeros((2,len))
x_l1 = np.zeros((2,len))
x_l2 = np.zeros((2,len))




for i in range(len):
	temp1 = D+lam_1[i]*(P-D)
	x_DP[:,i] =temp1.T 
	temp2 = l1a+lam_1[i]*(l1b-l1a)
	x_l1[:,i] =temp2.T 
	temp3 = l2a+lam_1[i]*(l2b-l2a)
	x_l2[:,i] =temp3.T 
	
plt.plot(x_DP[0,:] , x_DP[1,:] ,label= '$Tangent$' )
plt.plot(x_l1[0,:] , x_l1[1,:] ,label= '(2   1)x = 3' )
plt.plot(x_l2[0,:] , x_l2[1,:] ,labeL= '(1  -1)x = 1 ' )


plt.plot(D[0] , D[1] , 'o')
plt.text(D[0]*(1+0.1) , D[1]*(1-0.1) , 'D')




plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')

r = np.linalg.norm(C-D)
theta = np.linspace(0,2*(np.pi),10000)
x1 = C[0] + (r*np.cos(theta))
x2 = C[1] + (r*np.sin(theta))  
plt.plot(x1 , x2)
plt.axis("equal")

plt.text(D[0]*(1+0.1) , D[1]*(1-0.1) , '(1,-1)')
plt.plot(C[0] , C[1] , 'o')
plt.text(C[0]*((4/3)+0.001) , C[1]*((1/3)-0.001) , '(4/3,1/3)')
plt.xlim(-0.5,0.5)
plt.ylim(-2,2)

plt.grid()
plt.show()