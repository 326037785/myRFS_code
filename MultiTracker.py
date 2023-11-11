import scipy.stats
import numpy as np
from numpy import newaxis as na
import joblib
from scipy.optimize import linear_sum_assignment
class Multitracker:
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
    
    def PHD_filter(self,density,model,meas):
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
            w_predict = w_update*model.P_S
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
            L_posterior = len(w_update)
            w_update,x_update,P_update = density.gaussian_prune(w_update,x_update,P_update,self.elim_threshold)
            L_prune = len(w_update)
            w_update,x_update,P_update = density.gaussian_merge(w_update,x_update,P_update,self.merge_threshold)
            L_merge = len(w_update)
            w_update,x_update,P_update = density.gaussian_cap(w_update,x_update,P_update,self.L_max)
            L_cap =len(w_update)

            # targets extraction
            for i in (w_update > 0.5).nonzero()[0]:
                num_targets = int(np.round(w_update[i]))
                # TODO: right now, appending arrays with varying # columns to a list => hard to plot
                est['X'][k].append(np.tile(x_update[..., i, na], num_targets))
                est['N'][k] += num_targets

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
    def cmbm_filter(self,density,model,meas):
        # frompyfunc的作用是生成一个普遍的函数ufunc，其中，括号里的第一项是要传递的函数，第二项是输入的个数，第三项是输出的个数
        # 整个表达式相当于先用 np生成了array，再给array赋值list
        est ={
            'X':np.frompyfunc(list,0,1)(np.empty((meas['K'],),dtype = object)),
            'N':np.zeros((meas['K'],))
        }
        meas['z_gated'] = list()

        # initial prior
        w_update = np.array([1.0])
        x_update = np.array([[0.1,0,0.1,0]]).T
        P_update = np.expand_dims(np.diag([1, 1, 1, 1]) ** 2, axis=-1)  # add axis to the end -> (4,4,1)
        r_update = np.array([0.1])
        T_update = 1
        for k in range(meas['K']):
            r_predict = np.empty((T_update,))
            w_predict = np.empty((T_update,))
            x_predict = np.empty((model.x_dim,T_update))
            P_predict = np.empty((model.x_dim,model.x_dim,T_update))
            # predict
            x_predict, P_predict = density.kf_predict(model,w_update,x_update,P_update)
            w_predict = w_update
            r_predict = r_update*model.P_S
            w_predict = np.concatenate((model.w_birth, w_predict))
            r_predict = np.concatenate((model.r_birth,r_predict))
            x_predict = np.concatenate((model.m_birth, x_predict), axis=-1)
            P_predict = np.concatenate((model.P_birth, P_predict), axis=-1)
            T_predict = model.T_birth+T_update
            L_predict = np.ones((T_predict,),dtype= int)
            r_predict = density.range_r(r_predict)

            # construct for pseudo PHD for update
            L_pseudo = L_predict.sum()                                         # number of Gaussians in pseudo-PHD
            x_pseudo = np.zeros((model.x_dim, L_pseudo))                         # means of Gaussians in pseudo-PHD
            P_pseudo = np.zeros((model.x_dim, model.x_dim, L_pseudo))            # covs of Gaussians in pseudo-PHD
            w_pseudo = np.zeros((L_pseudo, 1))                                   # weights of Gaussians in pseudo-PHD
            w_pseudo1 = np.zeros((L_pseudo, 1))                                  # alt weight (1) of Gaussians in pseudo-PHD - used in CB-MeMBer update later
            w_pseudo2 = np.zeros((L_pseudo, 1))
            start_pt = 0
            for t in range(T_predict):
                end_pt = start_pt + L_predict[t] - 1
                if end_pt==start_pt:
                    x_pseudo[:, t] = x_predict[:, t]
                    x_pseudo[:, t] = x_predict[:, t]
                    P_pseudo[..., t] = P_predict[..., t]
                    w_pseudo[t] = r_predict[t] / (1 - r_predict[t]) * w_predict[t]
                    w_pseudo1[t] = r_predict[t] / (1 - r_predict[t] * model.P_D) * w_predict[t]
                    w_pseudo2[t] = r_predict[t] * (1 - r_predict[t]) / ((1 - r_predict[t] * model.P_D) ** 2) * w_predict[t]
                else:
                    x_pseudo[:, start_pt:end_pt] = x_predict[:, t]
                    P_pseudo[..., start_pt:end_pt] = P_predict[..., t]
                    w_pseudo[start_pt:end_pt] = r_predict[t] / (1 - r_predict[t]) * w_predict[t]
                    w_pseudo1[start_pt:end_pt] = r_predict[t] / (1 - r_predict[t] * model.P_D) * w_predict[t]
                    w_pseudo2[start_pt:end_pt] = r_predict[t] * (1 - r_predict[t]) / ((1 - r_predict[t] * model.P_D) ** 2) * w_predict[t]
                start_pt = end_pt + 1
            # gating
            if self.gate:
                z_gated =density.gate_meas_gms(model,meas['Z'][k],x_predict,P_predict,self.gamma)
                meas['z_gated'].append(z_gated)
            #update


        return est