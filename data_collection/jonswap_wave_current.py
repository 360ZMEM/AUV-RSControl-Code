import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

def jonswap_spectrum(f, alpha, g, f_p, gamma):
    sigma = np.where(f <= f_p, 0.07, 0.09)
    f = np.clip(f, 1e-10, None) 
    exponent = -((f - f_p)**2)/(2*(sigma**2)*f_p**2)
    S = (alpha * g**2 / ((2*np.pi)**4 * f**5)) * \
        np.exp(-5/4*(f_p/f)**4) * \
        gamma**np.exp(exponent)
    return S

def direction_distribution(theta, theta0=0):
    theta_diff = theta - theta0
    theta_diff = (theta_diff + np.pi) % (2*np.pi) - np.pi
    mask = np.abs(theta_diff) <= np.pi/2
    D = np.zeros_like(theta)
    D[mask] = (2/np.pi) * np.cos(theta_diff[mask])**2
    return D

def solve_k(omega, h, g=9.81, tol=1e-6, max_iter=100):
    k = omega**2/g
    for _ in range(max_iter):
        tanh_kh = np.tanh(k*h)
        f = g*k*tanh_kh - omega**2
        dfdk = g*tanh_kh + g*k*h*(1 - tanh_kh**2)
        delta = f / (dfdk + 1e-12)
        k -= delta
        if np.max(np.abs(delta)) < tol:
            break
    return k

alpha = 0.01
gamma = 3.3
f_p = 0.1

g = 9.81
theta0 = 0
t = 5
h = 45

n_freq, n_dir = 10, 10
f = np.linspace(0.05, 3*f_p, n_freq)
theta = np.linspace(-np.pi/2, np.pi/2, n_dir)
F, Theta = np.meshgrid(f, theta, indexing='ij')

S_f = jonswap_spectrum(F, alpha, g, f_p, gamma)
D_theta = direction_distribution(Theta, theta0)
S_total = S_f * D_theta

a = np.sqrt(2 * S_total * (f[1]-f[0]) * (theta[1]-theta[0]))

omega = 2*np.pi*F
k = solve_k(omega, h)

a_ext = a[..., np.newaxis, np.newaxis]
k_ext = k[..., np.newaxis, np.newaxis]
omega_ext = omega[..., np.newaxis, np.newaxis]
theta_ext = Theta[..., np.newaxis, np.newaxis]

kx = k_ext * np.cos(theta_ext)
ky = k_ext * np.sin(theta_ext)

def calculate_wave(x_max,y_max, t, x_min=0,y_min=0,resolution = 1.5, rand_phase = None):
    if type(rand_phase) == type(None):
        rand_phase = np.random.rand(*F.shape) * 2*np.pi
    phi_ext = rand_phase[..., np.newaxis, np.newaxis]
    Nx = int((x_max - x_min) / resolution)
    Ny = int((y_max - y_min) / resolution)
    x = np.linspace(x_min, x_max, Nx)
    y = np.linspace(y_min,y_max,Ny)
    X, Y = np.meshgrid(x, y)

    phase = kx*X + ky*Y - omega_ext*t + phi_ext

    eta = np.sum(a_ext * np.cos(phase), axis=(0,1)) 

    return eta, phase


def calculate_current(phase, z,h=30):
    if type(h) == np.ndarray:
        shape_x, shape_y = phase.shape[-2:]
        shape_hx, shape_hy = h.shape
        zoom_factor = (shape_x / shape_hx, shape_y / shape_hy)
        h = zoom(h, zoom_factor, order=1)
        h = h[np.newaxis, np.newaxis, ...]
    kh = k_ext * h
    coeff = a_ext * omega_ext * np.cosh(k_ext*(-z+h)) / np.sinh(kh + 1e-12)
    u = np.sum(coeff * np.cos(theta_ext) * np.cos(phase), axis=(0,1))
    v = np.sum(coeff * np.sin(theta_ext) * np.cos(phase), axis=(0,1))
    return u,v

constrain = lambda x_min, x, x_max: x if (x_min <= x <= x_max) else (x_min if (x < x_min) else x_max)


def get_cur_idx(x_max,y_max,x,y,x_min=0,y_min=0, resolution = 1.5):
    x = constrain(x_min,x,x_max - 0.1); y = constrain(y_min,y,y_max - 0.1)
    Nx = int ((x_max - x_min) / resolution)
    Ny = int((y_max - y_min) / resolution)
    x_grid = np.linspace(x_min, x_max, Nx)
    y_grid = np.linspace(y_min,y_max,Ny)
    x_idx, y_idx = np.where(x >= x_grid)[0][-1], np.where(y >= y_grid)[0][-1]
    x_resid, y_resid = x - x_grid[x_idx], y-y_grid[y_idx]
    return y_idx,x_idx,y_resid,x_resid

def interpolate(u_11,u_12,u_21,u_22,x_resid,y_resid,resolution):
    u1 = (resolution-x_resid)/resolution * u_11 + x_resid/resolution * u_12
    u2 = (resolution-x_resid)/resolution * u_21 + x_resid/resolution * u_22
    return (resolution-y_resid)/resolution * u1 + y_resid/resolution * u2

def get_pointwave(Z,x_max,y_max,x,y,x_min=0,y_min=0, resolution = 1.5):
    x_idx,y_idx,x_resid,y_resid = get_cur_idx(x_max,y_max,x,y,x_min,y_min,resolution=resolution)
    return interpolate(Z[x_idx,y_idx],Z[x_idx+1,y_idx],Z[x_idx,y_idx+1],Z[x_idx+1,y_idx+1],x_resid,y_resid,resolution)

def get_pointcurrent(U,V,x_max,y_max,x,y,x_min=0,y_min=0, resolution = 1.5):
    x_idx,y_idx,x_resid,y_resid = get_cur_idx(x_max,y_max,x,y,x_min,y_min,resolution=resolution)
    u = interpolate(U[x_idx,y_idx],U[x_idx+1,y_idx],U[x_idx,y_idx+1],U[x_idx+1,y_idx+1],x_resid,y_resid,resolution)
    v = interpolate(V[x_idx,y_idx],V[x_idx+1,y_idx],V[x_idx,y_idx+1],V[x_idx+1,y_idx+1],x_resid,y_resid,resolution)
    return u,v