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
import data_collection.jonswap_wave_current as wave
import terrain
import trimesh
import joblib
from matplotlib.colors import Normalize
from matplotlib.legend_handler import HandlerTuple
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from python_vehicle_simulator.vehicles import * 
from python_vehicle_simulator.lib import *
from utils_ import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--preview', type=int, default=-1, help='Preview frame NO.')
parser.add_argument('--fname', type=str, default='test', help='Filename of the record object')
parser.add_argument('--mode', type=str, default='train', help='Type of record object')
parser.add_argument('--speed', type=int, default=5, help='Video speed (x)')
args = parser.parse_args()


ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT_PATH)
if not os.path.exists('temp_images'):
    os.makedirs('temp_images')

dirname = 'record_object' + args.mode
fname = args.fname
PREVIEW_NUM = args.preview

# load file
with open(f'{dirname}/{fname}.pkl','rb') as f:
    record_object:Record_object = pickle.load(f)

ep_len = len(record_object.eta_AUV_record[0])
step_size = record_object.step_size
SP = args.speed
num_frames = int(ep_len / (SP))
border = record_object.border[:2]

# load 3D model
AUV_mesh = trimesh.load('resources/AUV.obj',force='mesh'); USV_mesh = trimesh.load('resources/USV.obj',force='mesh')
AUV_meshv, AUV_meshf, USV_meshv, USV_meshf = AUV_mesh.vertices, AUV_mesh.faces, USV_mesh.vertices, USV_mesh.faces

rotation_matrix = euler_rotation_matrix([0,np.pi / 2,0])

AUV_meshv = AUV_meshv @ rotation_matrix.T 

rotation_matrix = euler_rotation_matrix([np.pi / 2,0,np.pi])

USV_meshv = USV_meshv @ rotation_matrix.T 

ratio_Y = border[1] / border[0]
ratio_Z = 72 / border[0]  # for proper z-axis ratio
AUV_meshv *= np.array([1,1,ratio_Z])

RATIO_AUV = 7.2; RATIO_USV = 0.0009 * 8 # for proper AUV/USV size
AUV_meshv *= RATIO_AUV; USV_meshv *= RATIO_USV

xy = np.array(record_object.xy_record)

xy_color = ['#006FBF','#BF006F','#002222'] 

N_AUV = xy.shape[1]

ang_auv = []
xyz_asv = np.array(record_object.pos_usv)
sp = xyz_asv.shape[0]
W = 100 # smoothed traj.
xyz_asv[:,0] = np.array([np.mean(xyz_asv[i:i+W,0]) for i in range(sp)])
xyz_asv[:,1] = np.array([np.mean(xyz_asv[i:i+W,1]) for i in range(sp)])

hov_sidx = [[] for _ in range(N_AUV)]

for i in range(N_AUV):
    ang_auv.append(get_angle_from_traj(xy[:,i,:]))

ang_asv = get_angle_from_traj(xyz_asv) # get yaw angle for illustration

try:
    rand_phase = record_object.rand_phase
except:
    rand_phase = np.random.rand(wave.n_freq, wave.n_dir) * 2*np.pi

start_time = time.time()

def gen_pic(frame_idx):
    fig = plt.figure(figsize=(5.5, 4.5))
    ax = fig.add_subplot(1,1,1, projection='3d')
    ax.set_xlabel('x / m')
    ax.set_ylabel('y / m')
    ax.set_zlabel('depth / m') 
    ax.set_title('Data Collection (t={:.0f}s)'.format(frame_idx*SP))
    if record_object.mode[0]:
        Z, phase = wave.calculate_wave(border[0], border[1], frame_idx * SP, resolution=1.5, rand_phase=rand_phase)
        Nx = int((border[0] - 0) / 1.5)
        Ny = int((border[1] - 0) / 1.5)
        X_ = np.linspace(0, border[0], Nx); Y_ = np.linspace(0, border[0], Ny)
        X,Y = np.meshgrid(X_,Y_)

    surf2 = ax.plot_surface(X, Y, Z, linewidth=0, alpha=.6, cmap='ocean', vmin = -10, vmax = 8)
    tr = terrain.terrain; x_tr = tr.shape[0]; y_tr = tr.shape[1];
    X_tr = np.linspace(0, border[0], x_tr,endpoint=False); Y_tr = np.linspace(0, border[1], y_tr,endpoint=False)
    X_tr_, Y_tr_ = np.meshgrid(Y_tr, X_tr)
    surf = ax.plot_surface(Y_tr_, X_tr_, tr - 5, linewidth=0, alpha=.5, cmap='BrBG', vmin=tr.min(), vmax=-4) # ,antialiased=False

    scatter_color = ['#ff0080','#ff7f27','#a4d217','#2bbfff']; scatter_object = []
    sn_center = record_object.sn_center; sn_lamda = record_object.sn_lamda
    for idx, c in enumerate([4,6,9,12]):
        xy_sns = sn_center[np.array(sn_lamda) == c]
        so = ax.scatter(xy_sns[:,0], xy_sns[:,1],-xy_sns[:,2] + 2,marker='P',s=55,color=scatter_color[idx],edgecolors='k',linewidths=1.0,label='SNs')
        scatter_object.append(so)

    if record_object.mode[0]:
        draw_Z = np.array([10,20,30,40]) 

        X,Y,Z_ = np.meshgrid(X_,Y_,draw_Z)
        U, V, W = np.zeros_like(X), np.zeros_like(Y), np.zeros_like(Z_) 
        for idx, z in enumerate(draw_Z):
            U[:,:,idx], V[:,:,idx] = wave.calculate_current(phase, z, h=40)
        step_size = [18,18] 
        X = X[::step_size[0],::step_size[1],:]; Y = Y[::step_size[0],::step_size[1],:]; Z_ = Z_[::step_size[0],::step_size[1],:]
        U = U[::step_size[0],::step_size[1],:]; V = V[::step_size[0],::step_size[1],:]; W = W[::step_size[0],::step_size[1],:]
        speed = np.sqrt(U**2 + V**2 + W**2); speed_new = speed
        # speed_new = np.log10(speed); speed_new += (speed_new.min() + 1); speed_new /= (speed_new.max() / 13) 
        # U = U * speed_new / speed; V = V * speed_new / speed
        norm = Normalize(vmin=speed_new.min(), vmax=speed_new.max())
        colors = cm.cool(norm(speed_new.flatten()))
        quiver = ax.quiver(X.flatten(), Y.flatten(), -Z_.flatten(),
                        U.flatten(), V.flatten(), W.flatten(),
                        color=colors[:, :3], length=12, linewidths=1.2, arrow_length_ratio=0.36, alpha=0.6) # , antialiased=False

    eframe = frame_idx * SP


    xyz_asv[eframe,2] = wave.get_pointwave(Z, border[0], border[1], xyz_asv[eframe,0], xyz_asv[eframe,1]) * 0.25

    asv = ax.plot(xyz_asv[:eframe,0],xyz_asv[:eframe,1],xyz_asv[:eframe,2], lw=1.4, color=xy_color[-1], label=f'ASV')

    ax.scatter(xyz_asv[0,0],xyz_asv[0,1],xyz_asv[0,2],marker='>',s=55, color=xy_color[-1], edgecolors='k', linewidths=1.0)

    if record_object.mode[1]:

        USV_rotation_matrix = euler_rotation_matrix(ang_asv[:,eframe])

        USV_vertices = USV_meshv @ USV_rotation_matrix.T + xyz_asv[eframe] + np.array([0,0,1.99])

        mesh_color_USV = generate_mesh_colors(USV_vertices, USV_meshf,input_color=xy_color[-1],lum_ratio=0.899)

        ax.add_collection3d(Poly3DCollection(USV_vertices[USV_meshf], facecolors = mesh_color_USV, zorder=3)) 

    auv = []
    for j in range(N_AUV):

        auv.append(ax.plot(xy[:eframe,j,0],xy[:eframe,j,1],-xy[:eframe,j,2], lw=1.4, color=xy_color[j], label=f'AUV {j+1}'))

        ax.scatter(xy[0,j,0],xy[0,j,1],-xy[0,j,2],marker='>',s=55, color=xy_color[j], edgecolors='k', linewidths=1.0)

        AUV_rotation_matrix = euler_rotation_matrix(ang_auv[j][:,eframe])

        AUV_vertices = AUV_meshv @ AUV_rotation_matrix.T + xy[eframe,j] * np.array([1,1,-1])

        mesh_color_AUV = generate_mesh_colors(AUV_vertices, AUV_meshf,input_color=xy_color[j],lum_ratio=0.6)

        ax.add_collection3d(Poly3DCollection(AUV_vertices[AUV_meshf], facecolors = mesh_color_AUV)) # , antialiased=False

        hover_point = np.array(record_object.hover_point[j]); sidx = np.array(record_object.sidx[j])

        hover_point = xy[sidx[sidx <= eframe],j,:]

        if len(hover_point) != 0:
            ax.scatter(hover_point[:,0],hover_point[:,1],-hover_point[:,2],marker='o',s=180,color=xy_color[j],edgecolors=xy_color[j],alpha=0.45)


    ax.view_init(elev=1, azim=135)
    ax.set_zlim(-65,7)
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
        ax.legend(unique_handles, unique_labels,handler_map={tuple: HandlerTuple(ndivide=None)},loc='upper right')
    else:
        plt.axis('off')
    
    if frame_idx == PREVIEW_NUM:
        plt.show()
    plt.savefig(f'temp_images/frame_{frame_idx:04d}.png', dpi=224)

    if frame_idx % 8 == 0:
        print('frame {} / {} OK! time elapse {:.2f}s'.format(frame_idx, num_frames, (time.time() - start_time)), end ='\r')
    

    plt.close()

if PREVIEW_NUM >= 0:
    gen_pic(PREVIEW_NUM)
else:
    joblib.Parallel(n_jobs=8)(joblib.delayed(gen_pic)(i) for i in range(0,num_frames))

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
