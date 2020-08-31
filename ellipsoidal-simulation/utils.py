import numpy as np
import cv2
from scipy.ndimage.morphology import binary_dilation as imdilate


def rotation_matrix_3d(wx, wy, wz):
    R = np.array([
         [np.cos(wy) * np.cos(wz), -np.cos(wy) * np.sin(wz), np.sin(wy)],
         [np.sin(wx) * np.sin(wy) * np.cos(wz) + np.cos(wx) * np.sin(wz),
         -np.sin(wx) * np.sin(wy) * np.sin(wz) + np.cos(wx) * np.cos(wz),
         -np.sin(wx) * np.cos(wy)],
        [-np.cos(wx) * np.sin(wy) * np.cos(wz) + np.sin(wx) * np.sin(wz),
          np.cos(wx) * np.sin(wy) * np.sin(wz) + np.sin(wx) * np.cos(wz),
          np.cos(wx) * np.cos(wy)]
    ])
    return R

def funcf(m,K):
    x = np.ones((3,1))
    x[0] = m[0]
    x[1] = m[1]
    y = np.matmul(K,x)
    return y

def ellipsoid_simulation(I,K,SSe,f,abc,var_mu,xmax,negative):


    if negative:
        I = 255-I
    J = I.copy()
    (N,M) = I.shape
    

    R = np.zeros((N,M)) # ROI of simulated defect

    invK  = np.linalg.inv(K)

    if len(abc)==3: # elliposoid
        (a,b,c) = abc 
    else: # sphere
        a = abc 
        b = a
        c = a 

    # Computation of the 3 x 3 matrices Phi and L
    H       = np.linalg.inv(SSe)
    h0      = H[0,:]/a
    h1      = H[1,:]/b
    h2      = H[2,:]/c
    Hs      = np.zeros((3,3))
    Hs[:,0] = h0[0:3]
    Hs[:,1] = h1[0:3]
    Hs[:,2] = h2[0:3]
    hd      = np.zeros((3,1))
    hd[0]   = h0[3]
    hd[1]   = h1[3]
    hd[2]   = h2[3]
    Phi     = np.matmul(Hs,Hs.T)
    hhd     = np.matmul(hd,hd.T)
    hhd1    = 1-np.matmul(hd.T,hd)
    L       = np.matmul(np.matmul(Hs,hhd),Hs.T) + hhd1*Phi

    # Location of the superimposed area
    A       = L[0:2,0:2]
    mc      = np.array(-f*np.matmul(np.linalg.inv(A),L[0:2,2]))
    x       = np.linalg.eig(A)[0]
    C       = np.array([x[1],x[0]])
    la      = C
    a00     = np.linalg.det(L)/np.linalg.det(A)
    ae      = f*np.sqrt(-a00/la[0])
    be      = f*np.sqrt(-a00/la[1])
    al      = np.arctan2(C[1],C[0])+np.pi
    ra      = np.array( [ae*np.cos(al), ae*np.sin(al)] )
    rb      = np.array( [be*np.cos(al+np.pi/2), be*np.sin(al+np.pi/2)] )
    u1      = funcf(mc+ra,K)
    u2      = funcf(mc+rb,K)
    u3      = funcf(mc-ra,K)
    u4      = funcf(mc-rb,K)
    uc      = funcf(mc,K)
    e1      = u1+u2-uc
    e2      = u1+u4-uc
    e3      = u3+u2-uc
    e4      = u3+u4-uc
    Es      = np.concatenate((e1,e2,e3,e4),axis=1)
    E       = Es[0:2,:]
    Emax    = np.max(E,axis=1)
    Emin    = np.min(E,axis=1)
    umin    = int(np.fix(Emin[0]))
    umax    = int(np.fix(Emax[0]+1))
    vmin    = int(np.fix(Emin[1]))  
    vmax    = int(np.fix(Emax[1]+1))
    bb      = (vmin, vmax, umin, umax)

    if umin>=0 and umax<M and vmin>=0 and vmax<N: 
        q       = 255/(1-np.exp(var_mu*xmax))
        R[umin:umax,vmin:vmax] = 1
        R       = imdilate(R)
        z       = np.zeros((2,1))
        for u in range(umin,umax):
            z[0] = u
            for v in range(vmin,vmax):
                z[1] = v
                m = funcf(z,invK)
                m[0:2] = m[0:2]/f
                p = np.matmul(np.matmul(m.T,L),m)
                if p>0:
                    x = np.matmul(np.matmul(m.T,Phi),m)
                    d = 2*np.sqrt(p)*np.linalg.norm(m)/x
                    J[u,v] = np.exp(var_mu*d)*(I[u,v]-q)+q

    if negative:
        J = 255-J
    return J,bb


