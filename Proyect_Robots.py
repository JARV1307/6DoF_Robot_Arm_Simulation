#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'notebook')


# In[2]:


from roboticstoolbox import DHRobot, RevoluteDH , PrismaticDH
from math import pi
from spatialmath import SE3
import roboticstoolbox as rtb
#Then we set lenghs and mass of the joints
# link 1
m1 = 1
l1 = 0.670
# link 2
m2 = 1
l2 = 1.075
# link 3
m3 = 1
l3 = 0.225
# link 4
m4 = 1
l4 = 0.245
# link 5
m5 = 1
l5 = 1.280
# link 6
m6 = 1
l6 = 0.215

#Gravity
g = 9.81 

#Thenwe create the joints and put the limits of the joints given in the problem
#Thenwe create the joints and put the limits of the joints given in the problem
L1 = RevoluteDH(d=0.670,a=0.312, m=m1, alpha=-pi/2, qlim=[0, 136*pi/180])
L2 = RevoluteDH(a=1.075,d=0.030,alpha=pi,m=m2, qlim=[0, 312*pi/180])
L3 = RevoluteDH(a=-0.225,d=0,alpha=pi/2, m=m3, qlim=[0, 720*pi/180])
L4 = RevoluteDH(a=0,d=-1.280, m=m4, alpha=-pi/2, qlim=[0, 250*pi/180])
L5 = RevoluteDH(a=0,d=0, m=m5, alpha=pi/2, qlim=[0, 720*pi/180])
L6 = RevoluteDH(a=0,d=-0.215, m=m5, alpha=pi)

#Creation of the robot
robot = DHRobot([L1, L2, L3, L4, L5, L6], gravity=[0, g, 0], name="S2000") 
print(robot)


# In[3]:


robot.addconfiguration('qo',[0,-pi/2,pi,0,0,pi])
robot.addconfiguration('q2',[pi/2,-pi/4,2*pi/3,pi,-pi/2,pi])
robot.plot(robot.qo)


# In[4]:


traj = rtb.jtraj(robot.qo, robot.q2,50)
robot.plot(traj.q)


# In[5]:


rtb.qplot(traj.q)


# In[6]:


T1 = robot.fkine(robot.qo)
T2 = robot.fkine(robot.q2)
print(T1)


# In[7]:


print(T2)


# In[8]:


q_0=robot.ikine_LM(T1,ilimit=500,rlimit=100,search=True)


# In[9]:


print(traj.q)


# In[10]:


print(robot.ikine_LM(T1,q0=robot.qo,ilimit=800,rlimit=200,tol=1e-100,transpose=0.2)[0])


# In[11]:


print(robot.ikine_LM(T2,q0=robot.qo,ilimit=1000,rlimit=100,tol=1e-100,transpose=0.2))


# In[12]:


q3 = robot.ikine_LM(T1,q0=robot.qo,ilimit=800,rlimit=200,tol=1e-100,transpose=0.2)[0]
q4 = robot.ikine_LM(T2,q0=robot.qo,ilimit=1000,rlimit=100,tol=1e-100,transpose=0.2)[0]

#5th order polynomial interpolation
traj2 = rtb.jtraj(q3, q4, 60)
robot.plot(traj2.q)
#print(traj.q)


# In[13]:


#Aditional we present an spherical interpolation for rotation on the arm

import numpy as np
from pyquaternion import Quaternion

T6 = np.array([[0,0,1,1.807],[0,-1,0,0.03],[1,0,0,1.97],[0,0,0,1]])
T7 = np.array([[0,-1,0,-0.03],[-0.25881905,0,0.96592583,1.16585878],[-0.96592583,0,-0.25881905,0.07987435],[0,0,0,1]])

q6 = Quaternion(matrix=T6) 
q7 = Quaternion(matrix=T7)
qs = []
print(q6)
print()

print(q7)
print()

t6_1=Quaternion.slerp(q6, q7, 0).transformation_matrix
print(t6_1)
print()

t7_1=Quaternion.slerp(q6, q7, 1).transformation_matrix
print(t7_1)


# In[14]:


q_p  = robot.ikine_LM(SE3(Quaternion.slerp(q6, q7, 0).transformation_matrix)[0],q0=robot.qo,ilimit=1000,rlimit=100,tol=1e-100,transpose=0.2)[0]
print(q_p)
print()
q_p2 =robot.ikine_LM(SE3(Quaternion.slerp(q6, q7, 1).transformation_matrix)[0],q0=robot.qo,ilimit=1000,rlimit=100,tol=1e-100,transpose=0.2)[0]
print(q_p2)


# In[15]:


qs.append(robot.ikine_LM(T1,q0=robot.qo,ilimit=800,rlimit=200,tol=1e-100,transpose=0.2)[0])

for i in np.arange(0.1,0.9,0.1):
    q  = robot.ikine_LM(SE3(Quaternion.slerp(q6, q7, i).transformation_matrix)[0],q0=robot.qo,ilimit=1000,rlimit=100,tol=1e-100,transpose=0.2)[0]
    qs.append(q)
    
qf=robot.ikine_LM(T2,q0=robot.qo,ilimit=800,rlimit=200,tol=1e-100,transpose=0.2)[0]    
qs.append(qf)

robot.plot(np.array(qs))
#print(qs)


# In[16]:


import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.transformations as pt
import pytransform3d.trajectories as ptr
import pytransform3d.plot_utils as ppu



pose1 = np.copy(T6)
pose2 = np.copy(T7)
print(pose1)
dq1 = pt.dual_quaternion_from_transform(pose1)
dq2 = pt.dual_quaternion_from_transform(pose2)
Stheta1 = pt.exponential_coordinates_from_transform(pose1)
Stheta2 = pt.exponential_coordinates_from_transform(pose2)

n_steps = 60

# Screw linear interpolation of dual quaternions (ScLERP)
sclerp_interpolated_dqs = np.vstack([pt.dual_quaternion_sclerp(dq1, dq2, t) for t in np.linspace(0, 1, n_steps)])
sclerp_interpolated_poses_from_dqs = np.array([pt.transform_from_dual_quaternion(dq) for dq in sclerp_interpolated_dqs])


ax = pt.plot_transform(A2B=pose1, s=0.3, ax_s=2)
pt.plot_transform(A2B=pose2, s=0.3, ax=ax)


traj_from_dqs_sclerp = ppu.Trajectory(sclerp_interpolated_poses_from_dqs, s=0.1, c="b")
traj_from_dqs_sclerp.add_trajectory(ax)
plt.legend([traj_from_dqs_sclerp.trajectory],["Dual quaternion ScLERP"])
plt.show()


# In[17]:


sclerp_interpolated_poses_from_dqs #transformation matrices of each point


# In[18]:


position =[]
for matrix in sclerp_interpolated_poses_from_dqs:
    q  = robot.ikine_LM(SE3(matrix)[0],q0=robot.qo,ilimit=1000,rlimit=100,tol=1e-100,transpose=0.2)[0]
    position.append(q)


# In[19]:


robot.plot(np.array(position))


# In[20]:


rtb.qplot(np.array(position))


# In[21]:


i = 0
compar01 = []
compar23 = []
compar45 = []
for pos in position:
    compar01.append([np.array(traj2.q)[i][0],np.array(position)[i][0],np.array(traj2.q)[i][1],np.array(position)[i][1]])
    compar23.append([np.array(traj2.q)[i][2],np.array(position)[i][2],np.array(traj2.q)[i][3],np.array(position)[i][3]])
    compar45.append([np.array(traj2.q)[i][4],np.array(position)[i][4],np.array(traj2.q)[i][5],np.array(position)[i][5]])
    i=i+1


# In[22]:


rtb.qplot(np.array(compar01))


# In[23]:


rtb.qplot(np.array(compar23))


# In[24]:


rtb.qplot(np.array(compar45))


# In[42]:


#Another Way(Incomplete)
'''
from dual_quaternions import *
import math as ma

dq1 = DualQuaternion.from_homogeneous_matrix(T6)
dq2 = DualQuaternion.from_homogeneous_matrix(T7)

tau = np.arange(0,1,0.1)
prev = dq1.quaternion_conjugate()*dq2
l, m, theta, d = prev.screw()
c = []

for t in tau:
    p = np.array([ma.cos(t*theta/2), ma.sin(t*theta/2)*l])
    q = np.array([-1*t*d/2*ma.sin(t*theta/2), t*d/2*ma.cos(t*theta/2)*l])     ma.sin(t*theta/2*m)
    e = prev^t #Exponential is missing
    #Product 
    #c.append()
print(theta)
'''


# In[ ]:




