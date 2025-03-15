import numpy as np
import copy

class Record_object:
    def __init__(self, env, N_AUV = 2) -> None:
        self.N_AUV = N_AUV
        self.idu = np.zeros(self.N_AUV)
        self.N_DO = 0
        self.DQ = 0
        self.N_POI = env.N_POI
        self.FX = np.zeros(self.N_AUV)
        self.sum_rate = np.zeros(self.N_AUV)
        self.Ft = 0
        self.crash = 0
        self.ep_reward = np.zeros(self.N_AUV)
        self.Ec = np.zeros(self.N_AUV)
        self.Ht = np.zeros(self.N_AUV)
        self.xy_record = [] 
        self.hover_point = [[] for _ in range(self.N_AUV)]
        self.sidx = [[] for _ in range(self.N_AUV)]
        self.sn_center = env.SoPcenter
        self.sn_lamda = env.lda
        self.step_size = env.step_size
        self.border = env.border
        self.TD_loss = np.zeros(self.N_AUV); self.A_loss = np.zeros(self.N_AUV)
        self.wave = []
        self.eta_AUV_record = [[] for _ in range(self.N_AUV)]
        self.nu_AUV_record = [[] for _ in range(self.N_AUV)]
        self.rl_action_record = [[] for _ in range(self.N_AUV)]
        self.u_control_record = [[] for _ in range(self.N_AUV)]
        self.u_actual_record = [[] for _ in range(self.N_AUV)]
        self.pos_usv = []
        self.rand_phase = None
        self.mode = [True, True] # considering wave, considering USBL



    def update_metric(self, env, data_rate, hover, td_loss, a_loss) -> None:
        self.ep_reward += env.rewards / 10
        self.crash += env.crash
        # self.idu += idu # 
        self.sum_rate += data_rate
        self.Ft += self.step_size
        self.N_DO += env.N_DO
        self.FX += env.FX
        self.DQ += sum(env.b_S/env.Fully_buffer)
        self.Ec += env.ec
        self.TD_loss += td_loss; self.A_loss += a_loss
        self.xy_record.append(copy.deepcopy(env.xy))
        for i in range(self.N_AUV):
            if hover[i]: 
                self.hover_point[i].append(copy.deepcopy(env.xy[i]))
                self.sidx[i].append(int(self.Ft // self.step_size))
            self.eta_AUV_record[i].append(copy.deepcopy(env.eta_AUV[i]))
            self.nu_AUV_record[i].append(copy.deepcopy(env.nu_AUV[i]))
            self.u_actual_record[i].append(copy.deepcopy(env.u_actual_AUV[i]))
            self.u_control_record[i].append(copy.deepcopy(env.u_control[i]))
            self.rl_action_record[i].append(copy.deepcopy(env.action[i]))
        self.wave.append([env.u_h, env.v_h]) 
        self.pos_usv.append(copy.deepcopy(env.xy_usv))
        self.mode = [env.simulate_wave, env.simulate_usbl]
                
    def print_metric(self, mode='Train'): 
        N_DO = self.N_DO / (self.Ft / self.step_size)
        DQ = self.DQ / (self.Ft * self.N_POI / self.step_size)
        Ec = np.sum(self.Ec / (self.Ft - self.Ht) * self.step_size) / self.N_AUV
        self.TD_loss /= (self.Ft / self.step_size); self.A_loss /= (self.Ft / self.step_size)
        return ('LEN {:.0f} | TD_loss {:.2f} | A_loss {:.2f} '.format(self.Ft,np.sum(self.TD_loss),np.sum(self.A_loss)) if mode == 'Train' else '') + 'ep_r {:.0f} | L_data {:.2f} | sum_rate {:.2f} | idu {} | ec {:.2f} | N_D {:.0f} | CS {} | FX {}'.format(np.sum(self.ep_reward),DQ,np.sum(self.sum_rate),self.idu,Ec,N_DO,self.crash,self.FX)
