import os, sys
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT_PATH)
sys.path.append(ROOT_PATH)
import math
from env import Env
import numpy as np
import argparse
import copy
from record import Record_object
from td3 import TD3
import time
import pickle
import gc
from tqdm import tqdm
import jonswap_wave_current as wave

import warnings

warnings.filterwarnings("error")

parser = argparse.ArgumentParser()
# ------ training paras ------
parser.add_argument('--is_train', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.0008)
parser.add_argument('--gamma', type=float, default=0.995)
parser.add_argument('--tau', type=float, default=0.001)
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--replay_capa', type=int, default=80000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--policy_freq', type=int, default=2)
parser.add_argument('--episode_num', type=int, default=1002)
parser.add_argument('--episode_length', type=int, default=1000, help='the length of an episode (sec)')
parser.add_argument('--save_model_freq', type=int, default=25)
parser.add_argument('--load_ep', type=int, default=300)
# ------ env paras ------
parser.add_argument('--R_dc', type=float, default=18., metavar='R_DC',help='the radius of data collection')
parser.add_argument('--border_x', type=float, default=200.,help='Area x size')
parser.add_argument('--border_y', type=float, default=200.,help='Area y size')
parser.add_argument('--border_z', type=float, default=200.,help='Area z size')
parser.add_argument('--wave_resolution', type=float, default=1.5,help='Min grid')
parser.add_argument('--n_s', type=int, default=60, help='The number of SNs')
parser.add_argument('--N_AUV', type=int, default=2, help='The number of AUVs')
parser.add_argument('--Q', type=float, default=2, help='Capacity of SNs (Mbits)')
parser.add_argument('--alpha', type=float, default=0.05, help='SNs choosing distance priority')
parser.add_argument('--step_size', type=float, default=1, help='Time per step(s)')
parser.add_argument('--sim_step_size', type=float, default=0.05, help='Time per simulation step (s). Make sure this value can divide into `--step_size`.')
parser.add_argument('--control_mode', type=str, default='Ssurface_nref_nv') 
parser.add_argument('--reward_ratio', type=float, default=1e4)
parser.add_argument('--k_yaw', type=str, default='[2, 1, 0.01]',help='Input list of length 2 (S-Surface) or 3 (PID)')
parser.add_argument('--k_depth', type=str, default='[1, 1, 0.01]',help='Input list of length 2 (S-Surface) or 3 (PID)')
parser.add_argument('--wave',action='store_true',help='Simualte JONSWAP wave and current.')
parser.add_argument('--usbl',action='store_true',help='Simualte USV-AUV collaboration and USBL positioning.')

args = parser.parse_args()

all_control_methods = {'SMC':'PVS original code implementation',
                       'Ssurface':'S surface controller (ref model)',
                       'Ssurface_nref':'S surface controller (w/o ref model)',
                       'Ssurface_nref_nv':'S surface controller (w/o ref model, not using velocity as Δe)',
                       'PID_nref':'PID controller (w/o ref model)',
                       'PID_nref_nv':'PID controller (w/o ref model, not using velocity as Δe)'}
if args.control_mode not in list(all_control_methods.keys()):
    raise NotImplementedError(f'All available controller are listed as below: \n {all_control_methods.__repr__()}')

try:
    k_yaw = eval(args.k_yaw); k_depth = eval(args.k_depth)
    if type(k_yaw) != list or type(k_depth) != list:
        raise TypeError
except:
    raise TypeError("Input `k_yaw` and `k_depth` type invalid.")

if len(k_yaw) < 3: k_yaw += [0] * (3 - len(k_yaw))
if len(k_depth) < 3: k_yaw += [0] * (3 - len(k_depth))

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = ROOT_PATH + "/models_ddpg/"
REC_PATH = ROOT_PATH + "/record_eval/"
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
if not os.path.exists(REC_PATH):
    os.makedirs(REC_PATH)

def eval():
    for ep in range(args.episode_num):
        state_c = env.reset()
        state = copy.deepcopy(state_c)
        record_object = Record_object(env, N_AUV)
        hovers = np.zeros(N_AUV)
        mode = np.zeros(N_AUV)
        ht = np.zeros(N_AUV)
        rand_phase = np.random.rand(wave.n_freq, wave.n_dir) * 2*np.pi
        for _ in tqdm(range(int(args.episode_length / args.step_size))):
            record_hover = np.zeros(N_AUV)
            td_loss = np.zeros(N_AUV); a_loss = np.zeros(N_AUV)
            act = []
            for i in range(N_AUV):
                iact = agents[i].select_action(state[i])
                act.append(iact)
            env.posit_change(act, hovers, rand_phase = rand_phase, control=args.control_mode)
            state_,rewards,Done,data_rate,ec,cs = env.step(hovers)
            for i in range(N_AUV):
                if mode[i] == 0:
                    agents[i].store_transition(state[i],act[i],rewards[i],state_[i],False)
                    state[i] = copy.deepcopy(state_[i])
                    if Done[i] == True:
                        record_object.idu[i] += 1
                        ht[i] = args.Q * env.updata[i] / data_rate[i] 
                        mode[i] += math.ceil(ht[i])
                        hovers[i] = True; record_hover[i] = True
                else:
                    mode[i] -= args.step_size
                    record_object.Ht[i] += args.step_size
                    if mode[i] == 0:
                        hovers[i] = False
                        state[i] = env.CHOOSE_AIM(idx=i,lamda=args.alpha)
            record_object.update_metric(env, data_rate, record_hover, td_loss, a_loss)
        record_object.rand_phase = rand_phase
        formatted_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        with open(REC_PATH + f'/EVAL_{ep}_TIME{formatted_time}.pkl', 'wb') as f:
            pickle.dump(record_object, f)

        print(f'\rEP {ep} | ', end='')
        print(record_object.print_metric(mode='Eval'))
        del record_object
        gc.collect()


if __name__ == "__main__":
    env = Env(args)
    N_AUV = args.N_AUV
    state_dim = env.state_dim
    action_dim = 3
    agents = [TD3(state_dim, action_dim) for _ in range(N_AUV)]
    for i in range(N_AUV):
        agents[i].load(SAVE_PATH, args.load_ep, idx=1)
    eval()