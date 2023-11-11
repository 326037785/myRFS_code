import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from numpy import newaxis as na
import joblib
from scipy.optimize import linear_sum_assignment
class Gaussian:
    @staticmethod
    def kf_update(model, m_pre, P_pre, z):
        # input:
        # model, a instanlized model containing all the object we need
        # m_pre , the predicted mean of PHD intensity with size = model.x_dim*n,n is the number of components
        # P_pre , the predicted covariance of ,,,,,,,,,,, with size= model.xdim*model.xdim*n
        # z, the gated measurement at time k
        n , m = m_pre.shape[1], z.shape[1] #get the number of components and measurements
        # 1. initilize or allocate space for the variables we'll use later

        # extract the imformation from our predictions as conterparts to measurment
        qz = np.empty((m,n)) #the likelihood matix
        x = np.empty((model.x_dim,m,n)) # the updated mean
        P = np.empty((model.x_dim,model.x_dim,n))
        I = np.eye(model.x_dim)
        for i in range(n):
            mz = model.H.dot(m_pre[...,i])#python在np的array中表示a(:,i)的方式
            S = model.H.dot(P_pre[...,i]).dot(model.H.T)+model.R
            S = (S+S.T)/2
            for j in range(m):
                qz[j,i] = scipy.stats.multivariate_normal.pdf(z[:,j],mz,S)
                # K= PH'inv(S), 相当于xA=b（xS=PH'),可以用sci库里的解线性方程
                #K = Pre[...,i].dot(model.H.T).dot(scipy.linalg.inv())
                # Kalman gain NOTE: doesn't cover semi-definite Pz
                # iPz = np.linalg.inv(Pz)
                # K_gain = P_predict[..., i].dot(self.model.H.T).dot(iPz)
                K = scipy.linalg.cho_solve(scipy.linalg.cho_factor(S),model.H.dot(P_pre[...,i])).T
                ## update
                x[...,i] = m_pre[...,i,None] + K.dot(z-mz[:,None])
                P[...,i] = (I-K.dot(model.H)).dot(P_pre[...,i])
        return qz, x, P
    
    def kf_predict(model,w, m_post,P_post):
        n = len(w)
        # preallocation
        m_pre = np.empty((model.x_dim,n))
        P_pre = np.empty((model.x_dim,model.x_dim,n))
        # predict
        for i in range(n):
            m_pre[...,i] = model.F.dot(m_post[...,i])
            P_pre[...,i] = model.F.dot(P_post[...,i]).dot(model.F.T)+model.Q
        return np.asarray(m_pre),np.asarray(P_pre)
    def gate_meas_gms(model,z,m,P,gamma):
        if z.shape[1] == 0:
            return z
        num = m.shape[1]
        gated = np.array([],dtype=np.int)
        for i in range(num):
            mz = model.H.dot(m[...,i])
            Pz = model.H.dot(P[...,i]).dot(model.H.T)+model.R
            # 马氏距离: dist = sqrt((z-mz)inv(P)(z-mz)) <=> dist = (sqrt(iP)(z-mz)) 这里卡方逆比较要取马氏距离的平方
            # chol(P)^2*dist = (z-mz)^2
            # In the context of a numpy array, None is used to add an extra dimension to the array. It is equivalent to using np.newaxis
            dz = scipy.linalg.solve_triangular(scipy.linalg.cholesky(Pz),(z-mz[:,None]))
            """np.union1d() is a function from the numpy library that returns the sorted, unique values that are in either of 
            the two input arrays. In this case, it is being used to find the union of two arrays gated 
            and np.where((dz**2).sum(axis=0) < self.gamma)  
            The resulting array is then assigned to gated """
            gated = np.union1d(gated,np.where((dz**2).sum(axis=0)< gamma))
        return z[:,gated]
    
    def gaussian_prune(w,m,P,thre):
        idx = np.asarray(w) > thre
        return w[idx], m[...,idx],P[...,idx]

    def gaussian_merge(w,m,P,thre):
        # this function is coded to merge the replicated or similar gaussian_components
        # w the weight of component
        # m mean
        # P covariance
        # thre threshold
        el = 0 # the final number of components
        # pre-allocation
        idx = np.arange(len(w))
        w_merged, m_merged, P_merged = [],[],[]
        while (len(idx)) !=0:
            w_i =w[idx]
            j = idx[np.argmax(w_i)] # get the idx with max weight in w_i
            # the orignial eqution dm = (x-x)'*inv(P)*(x-x) 
            #sqrt(P)*dm=(x-x)'?
            # idx na means newaxis
            # 这一步相当于把所有的分量和当前第j个分量进行对比，得到的dm相当于第j个分量于每个分量的距离，然后找出小于阈值的idx
            dm = scipy.linalg.solve_triangular(scipy.linalg.cholesky(P[..., j]), m[:, idx] - m[:, j, na])
            # This sums the squared values along axis=0. It calculates the sum of squared elements column-wise, resulting in a 2-dimensional array.
            cluttered_idx = idx[np.sum(dm**2,axis = 0)<=thre]
            # components to be merged
            w_el = w[cluttered_idx]
            x_el = m[...,cluttered_idx]
            P_el = P[...,cluttered_idx]
            w_merged.append(sum(w_el))
            # 把w提取为向量于x_el中的每行进行相乘再求列和
            m_merged.append((w_el[na, :]*x_el).sum(axis=1) / w_merged[el])
            P_merged.append((w_el[na,na,:]*P_el).sum(axis = -1)/w_merged[el])#-1 denote the last index
            idx =np.setdiff1d(idx,cluttered_idx)
            el += 1
            # np.asarray() converts a Python list, tuple, or an existing array into a NumPy array.
        return np.asarray(w_merged), np.asarray(m_merged).T,np.asarray(P_merged).T
    def gaussian_cap(w,m,P,num):
        if len(w)>num:
            idx = np.flip(np.argsort(np.asarray(w)))[: num] # 先选中w，然后提取出变量再逆序排列，最后提取前num个
            return w[idx], m[...,idx],P[...,idx]
        else:
            return w,m,P
    def range_r(R):
        i=0
        for r in R:
            if r>1:
                R[i] =0.999
            elif r<0.001:
                R[i]=0.001
            i += 1
        return R
    def esf(Z):
    # calculate elementary symmetric function using Mahler's recursive formula

        if len(Z) == 0:
            return np.array([1])

        n_z = len(Z)
        F = np.zeros((2, n_z))

        i_n = 1
        i_nminus = 0

        for n in range(n_z):
            F[i_n, 0] = F[i_nminus, 0] + Z[n]
            for k in range(2, n+2):
                if k == n + 1:
                    F[i_n, k-1] = Z[n] * F[i_nminus, k-2]
                else:
                    F[i_n, k-1] = F[i_nminus, k-1] + Z[n] * F[i_nminus, k-2]

            i_n, i_nminus = i_nminus, i_n

        s = np.concatenate(([1], F[i_nminus, :]))
        return s
    








