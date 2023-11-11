from matplotlib import pyplot as plt
import numpy as np
import time
import scipy.stats
from numpy import newaxis as na
import joblib
from scipy.optimize import linear_sum_assignment
class Model:
    def __init__(self,dt=1):
        self.dt  = dt
        self.x_dim = 4
        self.z_dim = 2
        self.F = np.array([[1,dt,0,0],
                           [0,1,0,0],
                           [0,0,1,dt],
                           [0,0,0,1]])
        self.B = np.array([[dt**2/2, 0],
                           [dt, 0],
                           [0, dt**2/2],
                           [0, dt]])
        self.sigmaV = 1
        self.Q = self.sigmaV**2*self.B.dot(self.B.T)
        R = np.asarray(np.diag([2,2]))
        self.R = np.multiply(R,R)
        self.H = np.array([[1,0,0,0],
                          [0,0,1,0]])
        self.P_S = 0.99
        self.Q_S = 1-self.P_S
        self.P_D = 0.68
        self.Q_D = 1-self.P_D

        ## possion birth RFS
        self.T_birth = 4
        self.L_birth = 4
        self.r_birth = 0.1*np.ones((self.L_birth,))
        self.w_birth = 0.03*np.ones((self.L_birth,))
        self.m_birth = np.zeros((self.x_dim,self.L_birth))
        self.B_birth = np.zeros((self.x_dim,self.x_dim,self.L_birth))
        self.P_birth = np.zeros((self.x_dim,self.x_dim,self.L_birth))

        self.m_birth[:,0] = np.array([0, 0, 0, 0]).astype(np.float)
        self.B_birth[...,0] = np.diag([10,10,10,10]).astype(np.float)
        self.P_birth[...,0] = self.B_birth[...,0].dot(self.B_birth[...,0].T)

        self.m_birth[:,1] = np.array([400, 0, -600, 0]).astype(np.float)
        self.B_birth[...,1] = np.diag([10,10,10,10]).astype(np.float)
        self.P_birth[...,1] = self.B_birth[...,1].dot(self.B_birth[...,1].T)

        self.m_birth[:,2] = np.array([-500, 0, -400, 0]).astype(np.float)
        self.B_birth[...,2] = np.diag([10,10,10,10]).astype(np.float)
        self.P_birth[...,2] = self.B_birth[...,2].dot(self.B_birth[...,2].T)

        self.m_birth[:,3] = np.array([-300, 0, 400, 0]).astype(np.float)
        self.B_birth[...,3] = np.diag([10,10,10,10]).astype(np.float)
        self.P_birth[...,3] = self.B_birth[...,3].dot(self.B_birth[...,3].T)

        self.lambda_c = 60
        self.range_c = np.array([[-1000,1000],
                                [-1000,1000]])
        self.pdf_c = 1/ np.prod(self.range_c[:,1] - self.range_c[:,0])

    def gen_truth(self):
        K = 100
        total_track = 12
        truth ={'K':K,
                'X':np.empty((self.x_dim,K,total_track))*np.nan,
                'N':np.zeros((K,),dtype=np.int8),
                'track_list':np.empty((K,total_track))*np.nan,
                'total_track':total_track
        }
        xstart =np.zeros((self.x_dim,total_track),dtype=np.float)
        xstart[:, 0] = np.array([0, 0, 0, -10])
        xstart[:, 1] = np.array([400, -10, -600, 5])
        xstart[:, 2] = np.array([-800, 20, -200, -5])
        xstart[:, 3] = np.array([400, -7, -600, -4])
        xstart[:, 4] = np.array([400, -2.5, -600, 10])
        xstart[:, 5] = np.array([0, 7.5, 0, -5])
        xstart[:, 6] = np.array([-800, 12, -200, 7])
        xstart[:, 7] = np.array([-200, 15, 800, -10])
        xstart[:, 8] = np.array([-800, 3, -200, 15])
        xstart[:, 9] = np.array([-200, -3, 800, -15])
        xstart[:, 10] = np.array([0, -20, 0, -15])
        xstart[:, 11] = np.array([-200, 15, 800, -5])
        # define birth and death time
        bd_time = np.array([[0,0,10,10,10,20,20,20,30,30,40,40],
                            [70,70,75,75,80,80,85,90,90,95,100,100]])
        for target_num in range(total_track):
            tstate = xstart[:,target_num]
            for k in range(bd_time[0,target_num],min(bd_time[1,target_num],K)):
                tstate =self.F.dot(tstate)+np.sqrt(self.Q).dot(np.random.randn(self.x_dim,))
                truth['X'][:,k,target_num] =tstate
                truth['track_list'][k,target_num]=target_num
                truth['N'][k] += 1
        return truth
    
    def gen_truth2(self):
        K = 100
        total_track = 1
        truth ={'K':K,
                'X':np.empty((self.x_dim,K,total_track))*np.nan,
                'N':np.zeros((K,),dtype=np.int8),
                'track_list':np.empty((K,total_track))*np.nan,
                'total_track':total_track
        }
        xstart =np.zeros((self.x_dim,total_track),dtype=np.float)
        xstart[:, 0] = np.array([400, 0, -400, -0])

        # define birth and death time
        bd_time = np.array([[0,],
                            [100]])
        for target_num in range(total_track):
            tstate = xstart[:,target_num]
            for k in range(bd_time[0,target_num],min(bd_time[1,target_num],K)):
                tstate =self.F.dot(tstate)+np.sqrt(self.Q).dot(np.random.randn(self.x_dim,))
                truth['X'][:,k,target_num] =tstate
                truth['track_list'][k,target_num]=target_num
                truth['N'][k] += 1
        return truth
    
    def gen_meas(self,truth):
        meas = {
            'K':truth['K'],
            'Z':[] #'Z': []  # np.empty((self.dim_obs, truth['K'], truth['total_tracks'])) * np.nan,
        }
        zero_mean = np.zeros((self.z_dim, ))
        for k in range(meas['K']):
            Z_k = None
            detected = np.random.rand(truth['N'][k],) <= self.P_D
            x = truth['X'][:,k,:]
            present_and_detected = ~ np.isnan(x.sum(axis=0)) # 互补符号， 1（第二位)的互补符号是-2,而0的互补符号是-1 但是在序列里至是给True/False取反
            #这里相当于 p_d = p_d & d,上一个语句只能说明state is present，下一个语句表明 stated is detected
            present_and_detected[present_and_detected == True] &= detected
            x = x[:,present_and_detected]
            # generated measurements and clutter
            r = np.random.multivariate_normal(zero_mean,self.R,size = x.shape[1]).T
            Z_k = self.H.dot(x)+r
            N_c = np.random.poisson(self.lambda_c)
            # [-1e3, 1e3;-1e3,1e3]*[-1;1]
            bounds = np.diag(self.range_c.dot(np.array([-1,1])))
            clutter = -1000.0+bounds.dot(np.random.rand(self.z_dim,N_c))
            # concatenate the sequence with column-wise
            Z_k = np.hstack((Z_k,clutter)) if Z_k is not None else clutter
            meas['Z'].append(Z_k)
        return meas

