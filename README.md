import numpy as np
import matplotlib.pyplot as plt

n = 3
A = np.matrix('0 1 0; 980 0 -2.8; 0 0 -100')
B = np.matrix('0; 0; 100')

K = np.matrix('365.46613439  21.77699344   0.49931265')

def plant(x,u):
    dx[0] = x[1]
    dx[1] = 980*x[0]-2.8*x[2]
    dx[2] = -100*x[2]+100*u
    return dx

def rk4(x,u,T):
    k1=plant(x,u)*T
    k2=plant(x+k1*0.5,u)*T
    k3=plant(x+k2*0.5,u)*T
    k4=plant(x+k3,u)*T
    dx = x + ((k1+k4)/6+(k2+k3)/3)
    return dx

x0 = np.array([[0.1],[-0.1],[-0.1]])
dx = np.zeros([3,1])
temp_s = np.zeros([3,1])
T=0.001
tf=1
sam=int(tf/T)
tspan = np.linspace(0,tf, sam+1)

xs=len(tspan)
x=np.zeros([n,xs])
u_sig=np.zeros([B.shape[1],xs])
x[0,0]=x0[0]
x[1,0]=x0[1]
x[2,0]=x0[2]

u_sig[:,0] = np.matmul(K,x[:,0])

for i in range(0, xs - 1):
    u = np.matmul(K, x[:, i])

    temp_s[0] = x[0, i]
    temp_s[1] = x[1, i]
    temp_s[2] = x[2, i]

    x_next = rk4(temp_s, u, T)
    x[0, i + 1] = x_next[0]
    x[1, i + 1] = x_next[1]
    x[2, i + 1] = x_next[2]

    u_sig[:, i + 1] = u

plt.figure()
plt.plot(tspan, x[0,:], label = "x1")
plt.plot(tspan, x[1,:], label = "x2")
plt.plot(tspan, x[2,:], label = "x3")
plt.grid()
plt.xlabel("Time")
plt.ylabel("State response")
plt.legend()
plt.show()

plt.figure()
plt.plot(tspan, u_sig[0,:], label = "u")
plt.grid()
plt.xlabel("Time")
plt.ylabel("Control signal")
plt.legend()
plt.show()
