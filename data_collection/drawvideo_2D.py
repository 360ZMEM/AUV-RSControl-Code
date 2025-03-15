import os, sys
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT_PATH)
sys.path.append(ROOT_PATH)
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from record import Record_object
import pickle
import cv2
import jonswap_wave_current as wave
import joblib
from matplotlib.colors import Normalize
from matplotlib.legend_handler import HandlerTuple
from python_vehicle_simulator.vehicles import * 
from python_vehicle_simulator.lib import *
from utils_ import *
import matplotlib.patheffects as path_effects


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--preview', type=int, default=-1, help='Preview frame NO.')
parser.add_argument('--fname', type=str, default='test', help='Filename of the record object')
parser.add_argument('--mode', type=str, default='train', help='Type of record object')
parser.add_argument('--speed', type=int, default=5, help='Video speed (x)')
args = parser.parse_args()


font = {'size': 8.3}
plt.rc('font', **font)

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT_PATH)
if not os.path.exists('temp_images'):
    os.makedirs('temp_images')

dirname = 'record_' + args.mode
fname = args.fname
PREVIEW_NUM = args.preview

with open(f'{dirname}/{fname}.pkl','rb') as f:
    record_object:Record_object = pickle.load(f)
ep_len = len(record_object.eta_AUV_record[0])
step_size = record_object.step_size
SP = args.speed
num_frames = int(ep_len / (SP))
border = record_object.border[:2]

xy = np.array(record_object.xy_record)

xy_color = ['#003070','#AF226F','#002222']

N_AUV = xy.shape[1]

try:
    rand_phase = record_object.rand_phase
except:
    rand_phase = np.random.rand(wave.n_freq, wave.n_dir) * 2*np.pi


# Generate figure

start_time = time.time()

def gen_pic(frame_idx):
    fig = plt.figure(figsize=(4.2, 3.8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('x / m')
    ax.set_ylabel('y / m')
    ax.set_title('Data Collection (t={:.0f}s)'.format(frame_idx*SP))
    if record_object.mode[0]:
        Z, phase = wave.calculate_wave(border[0], border[1], frame_idx * SP, resolution=1.5, rand_phase=rand_phase)
        Nx = int((border[0] - 0) / 1.5)
        Ny = int((border[1] - 0) / 1.5)
        X_ = np.linspace(0, border[0], Nx); Y_ = np.linspace(0, border[0], Ny)
        X,Y = np.meshgrid(X_,Y_)
        # levels = np.linspace(Z.min(),Z.max(),100)
        orig_cmap = plt.get_cmap('ocean')
        new_cmap = truncate_colormap(orig_cmap, 0.3, 0.9)
        surf2 = ax.pcolormesh(X, Y, Z, linewidth=0, alpha=.85, cmap=new_cmap)
        cbar = fig.colorbar(surf2,label='Wave Height (m)') # ,orientation='horizontal'
        cbar.mappable.set_clim(vmin=-5,vmax=5)
        # cbar.set_label('Z', rotation=270, labelpad=15)


    scatter_color = ['#ff0080','#ff7f27','#a4d217','#2bbfff']; scatter_object = []
    sn_center = record_object.sn_center; sn_lamda = record_object.sn_lamda
    for idx, c in enumerate([4,6,9,12]):
        xy_sns = sn_center[np.array(sn_lamda) == c]
        so = ax.scatter(xy_sns[:,0], xy_sns[:,1],marker='P',s=55,color=scatter_color[idx],edgecolors='k',linewidths=1.0,label='SNs',alpha=0.8)
        scatter_object.append(so)

    eframe = frame_idx * SP

    if record_object.mode[0]:
        z = 10
        X,Y = np.meshgrid(X_,Y_)
        U, V = np.zeros_like(X), np.zeros_like(Y) 
        U, V = wave.calculate_current(phase, z, h=40)
        step_size = [6,6] 
        X = X[::step_size[0],::step_size[1]]; Y = Y[::step_size[0],::step_size[1]]; 
        U = U[::step_size[0],::step_size[1]]; V = V[::step_size[0],::step_size[1]]; 
        speed = np.sqrt(U**2 + V**2); speed_new = speed
        norm = Normalize(vmin=speed_new.min(), vmax=speed_new.max())
        colors = cm.winter(norm(speed_new.flatten()))
        quiver = ax.quiver(X.flatten(), Y.flatten(), 
                        U.flatten(), V.flatten(),
                        color=colors[:, :3], linewidths=1.2,  alpha=0.8,scale_units='inches',width=0.003,scale=15) 


    auv = []
    
    j = 1
    auv.append(ax.plot(xy[:eframe,j,0],xy[:eframe,j,1], lw=1.99, color=xy_color[j], label=f'AUV'))

    ax.scatter(xy[0,j,0],xy[0,j,1],marker='>',s=99, color=xy_color[j], edgecolors='k', linewidths=1.0)

    ax.scatter(xy[eframe,j,0],xy[eframe,j,1],marker='o',s=99, color=xy_color[j], edgecolors='k', linewidths=1.0)

    hover_point = np.array(record_object.hover_point[j]); sidx = np.array(record_object.sidx[j]); hover_point_next = []

    if len(sidx) != 0:
        hover_point = xy[sidx[sidx <= eframe],j,:]

        hover_point_next = xy[sidx[sidx > eframe],j,:]

    alpha1 = int(frame_idx // 12) % 2
    alpha2 = (24 + 1.5 * (-12 * (alpha1) + (2 * alpha1 - 1) * (frame_idx % 12)) ) / 24


    for k in range(len(hover_point)):

        ax.text(hover_point[k,0]-2,hover_point[k,1]-2,f'{k+1}',color=xy_color[j],path_effects=[path_effects.Stroke(linewidth=1,foreground="white"),path_effects.Normal()])

    if len(hover_point) != 0:
        ax.scatter(hover_point[:,0],hover_point[:,1],marker='o',s=699,color=xy_color[j],edgecolors=xy_color[j],alpha=0.25)
    ax.scatter(300,300,marker='o',s=72,color=xy_color[j],edgecolors=xy_color[j],alpha=0.25,label='Waypoints')

    if len(hover_point_next) != 0:
        ax.scatter(hover_point_next[0,0],hover_point_next[0,1],marker='o',s=699,linestyle='--', edgecolors=xy_color[j],alpha=alpha2,c='None',linewidth=1.3)
    ax.scatter(300,300,marker='o',s=72,linestyle='--', edgecolors=xy_color[j],c='None',label='Next waypoint',linewidth=0.8)

    ax.set_xlim(0,border[0])
    ax.set_ylim(0,border[1])

    handles, labels = ax.get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    
    unique_handles[0] = tuple(handles[:4])
    DRAW_AXIS = True

    if DRAW_AXIS:
        ax.legend(unique_handles, unique_labels,handler_map={tuple: HandlerTuple(ndivide=None)},columnspacing=1.5,loc='lower left',prop = {'size':7.2})
    else:
        plt.axis('off')

    plt.tight_layout()
    if frame_idx == PREVIEW_NUM:
        plt.show()
    plt.savefig(f'temp_images/frame_{frame_idx:04d}.png', dpi=224) 

    if frame_idx % 8 == 0:
        print('frame {} / {} OK! time elapse {:.2f}s'.format(frame_idx, num_frames, (time.time() - start_time)), end ='\r')

    plt.close()



if PREVIEW_NUM >= 0:
    gen_pic(PREVIEW_NUM)
else:
    joblib.Parallel(n_jobs=8)(joblib.delayed(gen_pic)(i) for i in range(num_frames))

    first_frame = cv2.imread('temp_images/frame_0000.png')
    height, width, layers = first_frame.shape
    frame_size = (width, height)

    frame_rate = 20

    fourcc = cv2.VideoWriter_fourcc(*'X264')  
    video = cv2.VideoWriter('output_video.mp4', fourcc, frame_rate, frame_size)

    for i in range(num_frames):
        frame = cv2.imread(f'temp_images/frame_{i:04d}.png')
        video.write(frame)

    video.release()


    for i in range(num_frames):
        os.remove(f'temp_images/frame_{i:04d}.png')
    os.rmdir('temp_images')
