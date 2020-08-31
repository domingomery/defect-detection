# Simulation of ellipsoidal defects
#
# This method generates a 3D ellipoidal (in 3D space) and projects its onto
# projection plane using a perspective transformation. For details of the
# model, see:
#
# # Mery, D. (2001, November). A new algorithm for flaw simulation in castings 
# by superimposing projections of 3D models onto X-ray images. In SCCC 2001. 
# 21st International Conference of the Chilean Computer Science Society 
# (pp. 193-202). IEEE. https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=972648
#
# The ellipsoidals defects are generated randomly. This code does not check if
# the projected ellipsoid belongs to the casting or not. It only checks if the
# bounding box is inside the image.
#
# (c) D. Mery, XCV Lab, Universidad Catolica de Chile, 2020
# http://xcv.ing.uc.cl


import numpy as np
import matplotlib.pylab as plt
from utils import ellipsoid_simulation, rotation_matrix_3d
from cv2 import imread, IMREAD_GRAYSCALE
import random

# Original X-ray image with no defects

# Example 1: Castings
img_name = 'casting.jpg'
darker = False # for castings, defects are brighter than the test object

# Example 2: Welds
# img_name = 'weld.png'
# darker = True # for welds, defects are darker than the test object

I = np.double(imread(img_name, IMREAD_GRAYSCALE))

N,M = I.shape

ok = False

while not ok:
    # Transformation (X,Y,Z)->(Xb,Yb,Zb)
    wx = 2*np.pi*random.random()
    wy = 2*np.pi*random.random()
    wz = 2*np.pi*random.random()
    R1 = rotation_matrix_3d(wx,wy,wz)
    tx = -166+460*random.random()
    ty = -250+560*random.random()
    tz = 1000
    t1 = np.array([tx, ty, tz])
    S = np.vstack([np.hstack([R1, t1[:, np.newaxis]]), np.array([0, 0, 0, 1])])

    # Transformation (Xp,Yp,Zp)->(X,Y,Z)    
    R2 = rotation_matrix_3d(0,0,np.pi/3)
    t2 = np.array([0,0,0])
    Se = np.vstack([np.hstack([R2, t2[:, np.newaxis]]), np.array([0, 0, 0, 1])])

    # Transformation (Xp,Yp,Zp)->(Xb,Yb,Zb)   
    SSe = np.matmul(S,Se)

    # Transformation (x,y)->(u,v)
    K = np.array([[1.1, 0, 235], [0, 1.1, 305], [0,0,1]])

    # Dimensions of the ellipsoid in mm
    a = 1+10*random.random()
    b = 1+10*random.random()
    c = 1+10*random.random()

    # Focal distance in mm
    f = 1500

    # X-ray Absorption coefficient
    mu = 0.01 # the larger the starker the intensity of the defect

    # Maximal observable thickness in mm in the X-ray image
    xmax = 400


    # Simulation: J simulated image, bb bounding box
    J,bb = ellipsoid_simulation(I,K,SSe,f,(a,b,c),mu,xmax,darker)
    (xmin,xmax,ymin,ymax) = bb

    if xmin>=0 and xmax<M and ymin>=0 and ymax<N: 
        ok = True


print('Bounding Box: '+str(bb))

# Output
fig1, ax = plt.subplots(1, 2, figsize=(12,6))
ax[0].imshow(I, cmap='gray'), ax[0].axis('off')
ax[1].imshow(J, cmap='gray'), ax[1].axis('off')
ax[1].plot([xmin,xmin,xmax,xmax,xmin],[ymax, ymin, ymin, ymax, ymax]) 

plt.show()
