import os, sys
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT_PATH)
sys.path.append(ROOT_PATH)
import numpy as np
import math
import gc
import terrain
import jonswap_wave_current as wave
import usv_usbl
import importlib.util
from python_vehicle_simulator.vehicles import *
from python_vehicle_simulator.lib import *

class Env(object): 
    def __init__(self,args):
        # ---- paras args ----
        self.N_SNs = args.n_s
        self.N_AUV = args.N_AUV
        self.X_max = args.border_x
        self.Y_max = args.border_y
        self.Z_max = args.border_z
        self.border = np.array([self.X_max,self.Y_max,np.max(terrain.terrain)])
        self.r_dc = args.R_dc
        self.N_POI = args.n_s
        self.epi_len = args.episode_length
        self.wave_resolution = args.wave_resolution
        self.step_size = args.step_size
        self.sim_step_size = args.sim_step_size
        self.N_sim = int(self.step_size / self.sim_step_size)
        self.wave_amp = args.wave; self.simulate_usbl = args.usbl
        self.usbl = usv_usbl.USBL()
        self.reward_ok = True
        # NOTE NEW
        self.X_min = 0
        self.Y_min = 0
        self.r_dc = args.R_dc
        self.f = 20 # khz, AUV ~ SNs
        self.b = 1
        self.safe_dist = 10
        self.H = 100 # water depth
        self.V_max = 2.2
        self.V_min = 1.2
        self.S = 60
        self.P_u = 9e-2

        self.SoPcenter = np.zeros((self.N_POI, 3)) # center of SNs
        self.state_dim = 12 + 3 * (self.N_AUV - 1)
        self.state = [np.zeros(self.state_dim)] * self.N_AUV
        self.rewards = []
        self.xy = np.zeros((self.N_AUV, 3))
        self.xy_usv = np.zeros(3)
        self.obs_xy = np.zeros((self.N_AUV, 3))
        self.vxy = np.zeros((self.N_AUV, 3)) 
        self.dis = np.zeros((self.N_AUV, self.N_POI))
        self.dis_hor = np.zeros((self.N_AUV, self.N_POI)) # horizontal distance

        self.LDA = [4,6,9,12] # poisson variables
        CoLDA = np.random.randint(0, len(self.LDA), self.N_POI)
        self.lda = [self.LDA[CoLDA[i]] for i in range(self.N_POI)] # assign poisson 
        self.b_S = np.random.randint(0., 1000., self.N_POI).astype(np.float32) 
        self.Fully_buffer = 5000
        self.H_Data_overflow = [0] * self.N_AUV
        self.Q = np.array(
            [self.lda[i] * self.b_S[i] / self.Fully_buffer for i in range(self.N_POI)])
        self.idx_target = np.argsort(self.Q)[-self.N_AUV:]
        self.updata = self.b_S[self.idx_target] / self.Fully_buffer
        self.Ft = 0

        self.FX = np.zeros(self.N_AUV)
        self.ec = np.zeros(self.N_AUV)
        self.TL = np.zeros(self.N_AUV)
        self.N_DO = 0
        self.crash = np.zeros(self.N_AUV)
        self.u_h = np.zeros(self.N_AUV); self.v_h = np.zeros(self.N_AUV)
        self.AUV = [remus100('depthHeadingAutopilot',50,0,0,0,0) for _ in range(self.N_AUV)]
        self.eta_AUV = [np.array([0, 0, 0, 0, 0, 0], float) for i in range(self.N_AUV)]
        self.nu_AUV = [np.array([0, 0, 0, 0, 0, 0], float) for _ in range(self.N_AUV)]
        self.u_actual_AUV = [self.AUV[i].u_actual for i in range(self.N_AUV)]
        gc.collect()
        self.k1 = 100; self.k2 = 100
        self.action = None; self.u_control = [None for _ in range(self.N_AUV)]
        

    def calcRate(self,f,b,d,dir=0):
        f1 = (f-b/2) if dir == 0 else (f+b/2)
        lgNt = 17 - 30*math.log10(f1)
        lgNs = 40 + 26*math.log10(f1) - 60*math.log10(f+0.03)
        lgNw = 50 + 20*math.log10(f1) - 40*math.log10(f+0.4) 
        lgNth = -15 + 20*math.log10(f1)
        NL = 10 * math.log10(1000*b*(10**(lgNt/10)+10**(lgNs/10)+10**(lgNw/10)+10**(lgNth/10))) 
        alpha = 0.11*((f1**2)/(1+f1**2)) + 44*((f1**2)/(4100+f1**2)) + (2.75e-4)*(f1**2) + 0.003
        TL = 15 * math.log10(d) + alpha * (0.001 * d)
        SL = 10*math.log10(self.P_u) + 170.77
        R = 0.001 * b * math.log(1+10**(SL-TL-NL),2)
        return R
    
    def angprocess(self, ang):
        while not (ang <= np.pi and ang > -np.pi):
            if ang > np.pi:
                ang -= 2 * np.pi
            elif ang <= -np.pi:
                ang += 2 * np.pi
        return ang

    def get_state(self): # new func 
        if self.simulate_usbl:
            SEARCH_SPACE = 2
            tol = 1e-2 if self.Ft == 0 else 5e-2
            x_min = 0 if self.Ft == 0 else max(0, self.xy_usv[0] - SEARCH_SPACE)
            x_max = self.border[0] if self.Ft == 0 else min(self.border[0], self.xy_usv[0] + SEARCH_SPACE)
            y_min = 0 if self.Ft == 0 else max(0, self.xy_usv[1] - SEARCH_SPACE)
            y_max = self.border[1] if self.Ft == 0 else min(self.border[1], self.xy_usv[1] + SEARCH_SPACE)
            self.xy_usv[:2] = usv_usbl.calcposit_USV(bounds = [(x_min, x_max), (y_min, y_max)], tol=tol, pos_auv = self.xy)
        for i in range(self.N_AUV):
            state = []
            if self.simulate_usbl:
                usv_auv_diff = self.usbl.calcPosit(self.xy_usv - self.xy[i], idx = i)
                self.obs_xy[i] = self.xy_usv - usv_auv_diff
            else:
                self.obs_xy[i] = self.xy[i]
        # then get locs
        for i in range(self.N_AUV):
            state = []
            for j in range(self.N_AUV):
                if j == i:
                    continue
                state.append((self.obs_xy[j] - self.obs_xy[i]).flatten() / self.border) # 3 * (N-1)
            # posit Target SNs
            state.append((self.target_Pcenter[i] - self.obs_xy[i]).flatten() / self.border) # 3
            state.append((self.obs_xy[i]).flatten() / self.border) # 3
            # finally, FX and N_DO
            state.append([self.u_h[i],self.v_h[i]]) # 2
            terrain_dist = self.obs_xy[i,2] / terrain.get_terrain_height(self.X_max,self.Y_max,self.xy[i,0],self.xy[i,1])
            state.append([terrain_dist]) # 1
            state.append([self.angprocess(self.eta_AUV[i][5]) / np.pi])
            state.append([self.FX[i]/self.epi_len, self.N_DO / self.N_POI]) # 2
            self.state[i] = np.concatenate(tuple(state))
    
    def reset(self):
        self.FX = np.zeros(self.N_AUV)
        self.ec = np.zeros(self.N_AUV)
        self.TL = np.zeros(self.N_AUV)
        self.N_DO = 0
        self.crash = np.zeros(self.N_AUV)
        self.u_h = np.zeros(self.N_AUV); self.v_h = np.zeros(self.N_AUV)
        self.SoPcenter[:,0] = np.random.randint(self.safe_dist, self.X_max - self.safe_dist, size=self.N_POI)
        self.SoPcenter[:,1] = np.random.randint(self.safe_dist, self.Y_max - self.safe_dist, size=self.N_POI)
        self.SoPcenter[:,2] = np.array([terrain.get_terrain_height(self.X_max,self.Y_max,self.SoPcenter[i,0],self.SoPcenter[i,1]) for i in range(self.N_POI)]) 
        while True:
            dist_ok = True
            self.xy[:,0] = np.random.randint(self.safe_dist, self.X_max - self.safe_dist, size=self.N_AUV)
            self.xy[:,1] = np.random.randint(self.safe_dist, self.Y_max - self.safe_dist, size=self.N_AUV)
            for i in range(self.N_AUV):
                for j in range(i+1, self.N_AUV):
                    if np.linalg.norm(self.xy[i,:2] - self.xy[j,:2]) < 2 * self.safe_dist:
                        dist_ok = False
            if dist_ok == True:
                self.xy[:,2] = np.array([terrain.get_terrain_height(self.X_max,self.Y_max,self.xy[i,0],self.xy[i,1]) - 10 for i in range(self.N_AUV)]) 
                break
        self.b_S = np.random.randint(0, 1000, self.N_POI)

        # assign target SNs
        self.Q = np.array(
            [self.lda[i] * self.b_S[i] / self.Fully_buffer for i in range(self.N_POI)])
        self.idx_target = np.argsort(self.Q)[-self.N_AUV:]
        self.updata = self.b_S[self.idx_target] / self.Fully_buffer
        self.target_Pcenter = self.SoPcenter[self.idx_target]
        self.u_h = np.zeros(self.N_AUV); self.v_h = np.zeros(self.N_AUV)
        self.eta_AUV = [np.array([0, 0, 0, 0, 0, 0], float) for i in range(self.N_AUV)]
        for i in range(self.N_AUV):
            self.eta_AUV[i][:3] = self.xy[i]
        self.AUV = [remus100('depthHeadingAutopilot',50,0,0,0,0) for _ in range(self.N_AUV)]
        self.nu_AUV = [np.array([0, 0, 0, 0, 0, 0], float) for _ in range(self.N_AUV)]
        for i in range(self.N_AUV): self.AUV[i].z_d = self.xy[i,2]
        self.u_actual_AUV = [self.AUV[i].u_actual for i in range(self.N_AUV)]
        self.get_state()
        self.old_yaw = [0 for _ in range(self.N_AUV)]
        self.old_z = [self.xy[i,2] for _ in range(self.N_AUV)]
        return self.state
    
    def calc_AUV_terrain_dist(self, idx):
        base_dist = terrain.get_terrain_height(self.X_max,self.Y_max,self.xy[idx,0],self.xy[idx,1]) - self.xy[idx,2]
        return min(base_dist, self.xy[idx,2]), base_dist < self.xy[idx,2] 

    def calc_AUV_border_dist(self,idx):
        return np.array([min(self.xy[idx,0], self.X_max-self.xy[idx,0]),min(self.xy[idx,1], self.Y_max-self.xy[idx,1])])


    def posit_change(self,actions,hovers,k_yaw, k_depth,rand_phase = None,control='Ssurface'):
        self.action = actions
        if self.wave_amp:
            Z, phase = wave.calculate_wave(self.border[0], self.border[1], self.Ft, rand_phase=rand_phase, resolution=self.wave_resolution)

        for i in range(self.N_AUV):
            change_z = actions[i][0] * 0.8;
            yaw_rate = actions[i][1] * 0.15
            yaw = self.old_yaw[i] + yaw_rate
            z = self.old_z[i] + change_z
            self.old_yaw[i] = yaw; self.old_z[i] = z
            desired_propeller = (actions[i][2] + 3) * 0.5 * 760 # 760 - 1520
            terrain_dist, dir_ = self.calc_AUV_terrain_dist(i)
            self.AUV[i].ref_z = z
            self.AUV[i].ref_psi = yaw * (180 / np.pi)  
            self.AUV[i].ref_n = desired_propeller
            if self.wave_amp:
                # fix: constrain z to guarantee normal wave generation
                h = terrain.get_terrain_height(self.X_max,self.Y_max,self.xy[i,0],self.xy[i,1])
                U, V = wave.calculate_current(phase, constrain(0, self.xy[i, 2], h), h = h)
                U *= self.wave_amp; V*= self.wave_amp
                self.u_h[i], self.v_h[i] = wave.get_pointcurrent(U, V, self.border[0], self.border[1], self.xy[i,0], self.xy[i,1], resolution=self.wave_resolution)
                vel_h, beta_h = np.sqrt(self.u_h[i] ** 2 + self.v_h[i] ** 2), np.arctan2(self.v_h[i], self.u_h[i])
                self.AUV[i].V_c = vel_h; self.AUV[i].beta_c = beta_h
            for _ in range(self.N_sim):
                        
                u_control = self.AUV[i].depthHeadingAutopilot(self.eta_AUV[i],self.nu_AUV[i],self.sim_step_size,k_yaw=k_yaw, k_depth=k_depth, control=control)

                [self.nu_AUV[i], self.u_actual_AUV[i]] = self.AUV[i].dynamics(self.eta_AUV[i], self.nu_AUV[i], self.u_actual_AUV[i], u_control, self.sim_step_size)

                self.eta_AUV[i] = attitudeEuler(self.eta_AUV[i], self.nu_AUV[i], self.sim_step_size)

            self.u_control[i] = u_control

            self.xy[i] = self.eta_AUV[i][:3]

            terrain_dist, _ = self.calc_AUV_terrain_dist(i)
            self.FX[i] = (np.sum((self.xy[i][:2]) < 0) + np.sum((self.border[:2] - self.xy[i][:2]) < 0)) > 0

            FX = 0
            if terrain_dist <= 1:
                FX = 1
            self.FX[i] += (FX if self.FX[i] == 0 else 0)

            if hovers[i] == False:
                V = np.linalg.norm(self.nu_AUV[i][:3])
                F = (0.7 * self.S * (V**2)) / 2
                self.ec[i] = (F * V) / (-0.081*(V**3)+0.215*(V**2)-0.01*V+0.541) + 15
            else:
                self.ec[i] = 90 + 15



    def step(self,hovers):
        self.N_DO = 0
        self.b_S += [np.random.poisson(self.lda[i]) for i in range(self.N_POI)]
        for i in range(self.N_POI): # check data overflow
            if self.b_S[i] >= self.Fully_buffer:
                self.N_DO += 1
                self.b_S[i] = self.Fully_buffer
        self.updata = self.b_S[self.idx_target] / self.Fully_buffer
        self.crash = np.zeros(self.N_AUV)
        self.TL = np.zeros(self.N_AUV)
        self.rewards = np.zeros(self.N_AUV)
        data_rate = np.zeros(self.N_AUV)
        self.Ft += self.step_size
        self.get_state()
        # get crash information
        for i in range(self.N_AUV):
            for j in range(self.N_AUV):
                if j == i:
                    continue
                dxy = (self.xy[j]-self.xy[i]).flatten()
                sd = np.linalg.norm(dxy)
                if sd < 5:
                    self.crash[i] += 1
        # then calculating dis AUV ~ target SNs
            self.calc_dist(i)
            if self.dis[i, self.idx_target[i]] < self.r_dc:
                self.TL[i] = True
                data_rate[i] = max(self.calcRate(self.f,self.b,self.dis [i,self.idx_target[i]],0),self.calcRate(self.f,self.b,self.dis[i,self.idx_target[i]],1))
                self.b_S[self.idx_target[i]] = 0
            self.rewards = self.compute_reward()
        return self.state, self.rewards, self.TL, data_rate, self.ec, self.crash
    
    def calc_dist(self,idx):
        for i in range(self.N_POI):
            self.dis[idx][i] = math.sqrt(
                    pow(self.SoPcenter[i][0] - self.xy[idx][0], 2) + pow(self.SoPcenter[i][1] - self.xy[idx][1], 2) 
                    + pow(self.SoPcenter[i][2] - self.xy[idx][2], 2))
            self.dis_hor[idx][i] = math.sqrt(
                pow(self.SoPcenter[i][0] - self.xy[idx][0], 2) + pow(self.SoPcenter[i][1] - self.xy[idx][1], 2))
            
    def CHOOSE_AIM(self, idx=0, lamda=0.05):
        self.calc_dist(idx=idx)
        Q = np.array([self.lda[i] * self.b_S[i] / self.Fully_buffer - lamda * self.dis[idx][i] for i in range(self.N_POI)])
        idx_target = np.argsort(Q)[-self.N_AUV:]
        inter = np.intersect1d(idx_target,self.idx_target)
        if len(inter) < len(self.idx_target):
            diff = np.setdiff1d(idx_target,inter)
            self.idx_target[idx] = diff[0]
        else:
            idx_target = np.argsort(self.Q)[-(self.N_AUV+1):]
            self.idx_target[idx] = idx_target[0]
        self.target_Pcenter = self.SoPcenter[self.idx_target]
        # state[i]
        st_idx = 3 * (self.N_AUV - 1)
        self.state[idx][st_idx:st_idx+3] = (self.target_Pcenter[idx] - self.xy[idx]).flatten() / self.border
        self.state[idx][-2] = self.N_DO / self.N_POI
        return self.state[idx]
    
    def compute_reward(self): # oracle
        reward = np.zeros(self.N_AUV)
        # waypoint serving
        w_waypoint = 4000
        w_collision = 500
        w_border_xy = 6
        w_border_z = 12
        w_ec = 0.1

        for i in range(self.N_AUV):
            # distance to target waypoint
            dist_to_target = np.linalg.norm(self.xy[i] - self.target_Pcenter[i]) / np.linalg.norm(self.border)
            reward[i] += -w_waypoint * dist_to_target

            # distance between AUVs (for collision avoidance)
            for j in range(i+1,self.N_AUV):
                dist_between_auvs = np.linalg.norm(self.xy[j] - self.xy[i]) / np.linalg.norm(self.border)
                if dist_between_auvs < self.safe_dist:
                    reward[i] -= w_collision * ((self.safe_dist - dist_between_auvs))
            
            # target waypoint serving reward (Sparse)
            if self.TL[i] > 0:
                reward[i] += w_waypoint

            reward[i] -= w_ec * self.ec[i] # Energy consumption (Sparse)

            reward[i] -= w_border_xy * self.FX[i] # Penalty of crossing the border (Sparse)

            border_dist = self.calc_AUV_border_dist(i)
            for b in border_dist:
                if b < 1.5 * self.safe_dist:
                    reward[i] -= w_border_xy * ((1.5 * self.safe_dist - b)) # x / y constraint
                    
            terrain_dist, dir_  = self.calc_AUV_terrain_dist(i)
            height_dist = self.safe_dist if dir_ else 2 * self.safe_dist
            if terrain_dist < height_dist:
                reward[i] -= w_border_z * ((height_dist - terrain_dist)) # z constraint
        return reward / 100 # normalize
