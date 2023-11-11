import scipy.stats
import numpy as np
from numpy import newaxis as na
import joblib
from scipy.optimize import linear_sum_assignment
class Monotracker:
    def __init__(self,density,model):
        self.density = density
        self.P_G =0.999
        self.L_max =100
        self.elim_threshold = 1e-5
        self.merge_threshold = 4
        self.gamma = scipy.stats.chi2.ppf(self.P_G,model.z_dim) # self.gamma = scipy.stats.gamma.ppf(self.P_G, 0.5 * self.model.dim_obs, scale=2)
        self.gate = True
        self.flag =True
        self.F_EPS = np.finfo(float).eps
    def __str__(self):
        return f"{self.P_G} is just ok"
    def gaussian_sum_filter(self,density,model,meas):
        # frompyfunc的作用是生成一个普遍的函数ufunc，其中，括号里的第一项是要传递的函数，第二项是输入的个数，第三项是输出的个数
        # 整个表达式相当于先用 np生成了array，再给array赋值list
        est ={
            'X':np.frompyfunc(list,0,1)(np.empty((meas['K'],),dtype = object)),
            'N':np.zeros((meas['K'],))
        }
        meas['z_gated'] = list()

        # initial prior
        w_update = np.array([self.F_EPS])
        x_update = np.array([[0.1,0,0.1,0]]).T
        P_update = np.expand_dims(np.diag([1, 1, 1, 1]) ** 2, axis=-1)  # add axis to the end -> (4,4,1)
        L_update = 1
        for k in range(meas['K']):
            w_predict = np.empty((L_update,))
            x_predict = np.empty((model.x_dim,L_update))
            P_predict = np.empty((model.x_dim,model.x_dim,L_update))
            x_predict, P_predict = density.kf_predict(model,w_update,x_update,P_update)
            w_predict = w_update
            w_predict = np.concatenate((model.w_birth, w_predict))
            x_predict = np.concatenate((model.m_birth, x_predict), axis=-1)
            P_predict = np.concatenate((model.P_birth, P_predict), axis=-1)
            L_predict = model.L_birth+L_update
            # gating
            if self.gate:
                z_gated =density.gate_meas_gms(model,meas['Z'][k],x_predict,P_predict,self.gamma)
                meas['z_gated'].append(z_gated)
            # update
            z_num =z_gated.shape[1]
            # for missed_term
            x_update = x_predict.copy()
            P_update = P_predict.copy()
            w_update = model.Q_D*w_predict.copy()
            # update term
            if z_num>0:
                qz , x_temp, P_temp =density.kf_update(model,x_predict,P_predict,z_gated)
                for j in range(z_num):
                    w_temp = model.P_D*w_predict*qz[j,:]
                    w_temp = w_temp/(sum(w_temp)+model.lambda_c*model.pdf_c) 
                    x_update = np.concatenate((x_update,x_temp[:,j,:]),axis = -1)
                    P_update = np.concatenate((P_update,P_temp),axis=-1)
                    w_update = np.concatenate((w_update,w_temp))
            # prune, merge and cap
            w_update /= sum(w_update)
            L_posterior = len(w_update)
            w_update,x_update,P_update = density.gaussian_prune(w_update,x_update,P_update,self.elim_threshold)
            L_prune = len(w_update)
            w_update,x_update,P_update = density.gaussian_merge(w_update,x_update,P_update,self.merge_threshold)
            L_merge = len(w_update)
            w_update,x_update,P_update = density.gaussian_cap(w_update,x_update,P_update,self.L_max)
            L_cap =len(w_update)

            # targets extraction
            if len(w_update)>0:
                idx_max = np.argmax(np.asarray(w_update))
                print(idx_max)
                est['X'][k].append(x_update[...,idx_max,na])
                est['N'][k] += 1
            else:
                idx_max = 0
                #est['X'][k].append(x_update[...,idx_max,na])
                est['N'][k] += 0
            # DIAGNOSTICS
            if self.flag:
                print('time = {:3d} | '
                      'est_mean = {:4.2f} | '  # integral of the PHD == expected # targets in the whole state space
                      'est_card = {:3f} | '  # targets estimated by the filter (estimated cardinality of the state RFS)
                      'gm_orig = {:3d} | '
                      'gm_elim = {:3d} | '
                      'gm_merg = {:3d}'.format(k, sum(w_update), est['N'][k], L_posterior, L_prune, L_merge))
        # convert list of arrays to one 2D array
        # (each such array has varying # columns at each position in est['X'][k])
        for k in range(meas['K']):
            est['X'][k] = np.concatenate(est['X'][k], axis=1) if len(est['X'][k]) > 0 else None

        return est
    
    def bernoulli_gms_filter(self,density,model,meas):
        est ={
            'X':np.frompyfunc(list,0,1)(np.empty((meas['K'],),dtype = object)),
            'N':np.zeros((meas['K'],))
        }
        meas['z_gated'] = list()

        # initial prior
        w_update = np.array([1.0])
        x_update = np.array([[0.1,0,0.1,0]]).T
        P_update = np.expand_dims(np.diag([1, 1, 1, 1]) ** 2, axis=-1)  # add axis to the end -> (4,4,1)
        L_update = 1
        r_update = 0.1
        r_birth = model.r_birth[0]
        
        #predict
        for k in range(meas['K']):
            r_predict = r_birth*(1-r_update)+model.P_S*r_update
            w_predict = np.empty((L_update,))
            x_predict = np.empty((model.x_dim,L_update))
            P_predict = np.empty((model.x_dim,model.x_dim,L_update))
            x_predict, P_predict = density.kf_predict(model,w_update,x_update,P_update)
            w_predict = w_update
            w_predict = np.concatenate((model.w_birth, w_predict))
            x_predict = np.concatenate((model.m_birth, x_predict), axis=-1)
            P_predict = np.concatenate((model.P_birth, P_predict), axis=-1)
            L_predict = model.L_birth+L_update
                        # gating
            if self.gate:
                z_gated =density.gate_meas_gms(model,meas['Z'][k],x_predict,P_predict,self.gamma)
                meas['z_gated'].append(z_gated)
            # update
            z_num =z_gated.shape[1]
            # for missed_term
            x_update = x_predict.copy()
            P_update = P_predict.copy()
            w_update = model.Q_D*w_predict.copy()*(model.lambda_c*model.pdf_c)
            if z_num>0:
                qz , x_temp, P_temp =density.kf_update(model,x_predict,P_predict,z_gated)
                for j in range(z_num):
                    w_temp = model.P_D*w_predict*qz[j,:]
                    w_temp = w_temp/(sum(w_temp)+model.lambda_c*model.pdf_c) 
                    x_update = np.concatenate((x_update,x_temp[:,j,:]),axis = -1)
                    P_update = np.concatenate((P_update,P_temp),axis=-1)
                    w_update = np.concatenate((w_update,w_temp))
            # prune, merge and cap
            w_update /= sum(w_update)
            r_update = (r_predict*w_update.sum())/((1-r_predict)*sum(w_update)+model.lambda_c*model.pdf_c)
            r_update =density.range_r(r_update)
            L_posterior = len(w_update)
            w_update,x_update,P_update = density.gaussian_prune(w_update,x_update,P_update,self.elim_threshold)
            L_prune = len(w_update)
            w_update,x_update,P_update = density.gaussian_merge(w_update,x_update,P_update,self.merge_threshold)
            L_merge = len(w_update)
            w_update,x_update,P_update = density.gaussian_cap(w_update,x_update,P_update,self.L_max)
            L_cap =len(w_update)
            # targets extraction
            if r_update>0.5:
                idx_max = np.argmax(np.asarray(w_update))
                print(idx_max)
                est['X'][k].append(x_update[...,idx_max,na])
                est['N'][k] += 1
            else:
                idx_max = 0
                #est['X'][k].append(x_update[...,idx_max,na])
                est['N'][k] += 0
            # DIAGNOSTICS
            if self.flag:
                print('time = {:3d} | '
                      'est_mean = {:4.2f} | '  # integral of the PHD == expected # targets in the whole state space
                      'prob = {:4.3f} | '
                      'est_card = {:3f} | '  # targets estimated by the filter (estimated cardinality of the state RFS)
                      'gm_orig = {:3d} | '
                      'gm_elim = {:3d} | '
                      'gm_merg = {:3d}'.format(k, sum(w_update), r_update, est['N'][k], L_posterior, L_prune, L_merge))
        # convert list of arrays to one 2D array
        # (each such array has varying # columns at each position in est['X'][k])
        for k in range(meas['K']):
            est['X'][k] = np.concatenate(est['X'][k], axis=1) if len(est['X'][k]) > 0 else None

        return est





                


