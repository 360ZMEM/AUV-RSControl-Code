from matplotlib.colors import LightSource
import matplotlib.cm as cm
import numpy as np
import colour
from matplotlib.colors import LinearSegmentedColormap


def euler_rotation_matrix(angles):
    alpha, beta, gamma = angles
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(alpha), -np.sin(alpha)],
        [0, np.sin(alpha), np.cos(alpha)]
    ])
    Ry = np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)]
    ])
    Rz = np.array([
        [np.cos(gamma), -np.sin(gamma), 0],
        [np.sin(gamma), np.cos(gamma), 0],
        [0, 0, 1]
    ])
    return Rz @ Ry @ Rx

def calculate_face_normals(vertices, faces):
    v1 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    v2 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    return np.cross(v1, v2)

def generate_mesh_colors(vertices, faces, input_color, lum_ratio=0.5):
    face_normals = calculate_face_normals(vertices, faces)
    ls = LightSource(azdeg=135, altdeg=80)
    face_colors = ls.shade_normals(face_normals, fraction=1.0)
    face_colors = cm.gray(face_colors)
    color = [colour.Color(input_color) for _ in range(len(face_colors))]
    lum = colour.Color(input_color).get_luminance()
    [color[idx].set_luminance(lum_ratio * lum + (1-lum_ratio) * fc[0]) for idx, fc in enumerate(face_colors)]
    return [[c.get_red(), c.get_green(), c.get_blue(), 1] for c in color]

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = LinearSegmentedColormap.from_list(
        'truncated_{name}'.format(name=cmap.name),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def get_angle_from_traj(in_xyz):
    W = 3
    len_xyz = in_xyz.shape[0]
    yaw = []; pitch = []
    for i in range(W,len_xyz):
        pos_diff = in_xyz[i] - in_xyz[i-W]
        yaw.append(np.arctan2(pos_diff[1], pos_diff[0]))
        pitch.append(np.arctan2(-pos_diff[2], np.linalg.norm(pos_diff[:2])))
    yaw = [yaw[0]] * W + yaw; pitch = [pitch[0]] * W + pitch
    return np.vstack([[0]*len_xyz,pitch,yaw])
