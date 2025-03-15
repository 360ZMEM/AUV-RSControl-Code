import numpy as np
from scipy.optimize import differential_evolution

def calcnegdetJ_USV(pos_usv, pos_auv):  # pos_auv -> 3d, pos_usv -> 2d
    # constrain z = 0
    pos_usv_3d = np.zeros(3)
    pos_usv_3d[:2] = pos_usv
    pos_usv = pos_usv_3d
    S_i = np.zeros(pos_auv.shape[0])
    p_i = np.zeros(pos_auv.shape[0])
    A_i = np.zeros(pos_auv.shape[0])
    # we don't consider coeffs
    for i in range(pos_auv.shape[0]):
        S_i[i] = np.linalg.norm(pos_usv - pos_auv[i])
        p_i[i] = np.linalg.norm(pos_auv[i][:2])
        A_i[i] = (p_i[i] ** 4 - 2 * (S_i[i] ** 2) * (p_i[i] ** 2)) / (2 * (S_i[i] ** 6))
    det_J1 = np.sum(S_i ** (-2))
    det_J2 = np.sum(2 * A_i + S_i ** (-2))
    det_J3 = 0
    for i in range(pos_auv.shape[0]):
        for j in range(i + 1, pos_auv.shape[0]):
            vi = pos_auv[i][:2] - pos_usv[:2]
            vj = pos_auv[j][:2] - pos_usv[:2]
            sinij = np.linalg.norm(np.cross(vi, vj)) / (
                np.linalg.norm(vi) * np.linalg.norm(vj)
            )
            det_J3 += 4 * A_i[i] * A_i[j] * (sinij) ** 2
    # if any value is not reasonable, return 0
    if np.sum(np.isnan(np.array([det_J1, det_J2, det_J3]))) != 0:
        return 0
    else:
        return -(det_J1 * det_J2 + det_J3)
    
def calcposit_USV(bounds, tol, pos_auv):
    calc_detJ = lambda pos_usv: calcnegdetJ_USV(pos_usv, pos_auv)
    return differential_evolution(calc_detJ, bounds=bounds, tol=tol, maxiter=500).x

# USBL Simulation

class USBL:
    def __init__(self) -> None:
        # Preset comm. freq.
        self.f = np.array([1.2e4, 1.4e4, 1.6e4, 1.8e4])
        self.c = 1500
        # wavelength
        self.lamda = self.c / np.max(self.f)
        self.d = 0.4 * self.lamda
        # sampling freq.
        self.f0 = 2.016e6
        # cross-shape hydrophones array
        self.hyd_posit = np.array(
            [
                [self.d / 2, 0, 0],
                [-self.d / 2, 0, 0],
                [0, self.d / 2, 0],
                [0, -self.d / 2, 0],
            ]
        )

    # calculating SN Ratio
    def calcSNR(self, f, b, d, format="active"):
        # sonar power
        SL = 145
        lgNt = 17 - 30 * np.log10(f)
        lgNs = 40 + 26 * np.log10(f) - 60 * np.log10(f + 0.03)
        lgNw = 50 + 20 * np.log10(f) - 40 * np.log10(f + 0.4)
        lgNth = -15 + 20 * np.log10(f)
        NL = 10 * np.log10(
            1000
            * b
            * (
                10 ** (lgNt / 10)
                + 10 ** (lgNs / 10)
                + 10 ** (lgNw / 10)
                + 10 ** (lgNth / 10)
            )
        )
        alpha = (
            0.11 * ((f**2) / (1 + f**2))
            + 44 * ((f**2) / (4100 + f**2))
            + (2.75e-4) * (f**2)
            + 0.003
        )
        TL = 15 * np.log10(d) + alpha * (0.001 * d)
        TS = 3
        if format == "active":
            SNR = SL - 2 * TL - NL + TS
        elif format == "passive":
            SNR = SL - TL + TS
        else:
            raise NotImplementedError
        return SNR

    # Measure the phase difference between the acoustic signal sent by sonar and the signal received with noise
    def get_phasedelay(self, dist, idx=0):  # idx -> AUV index
        # generate original singal, 10T
        t_length = int(10 * (self.f0 / self.f[idx]))
        t = np.arange(t_length) / self.f0
        # generate received signal,
        real_det_t = dist / self.c
        recv_signal = np.sin(2 * np.pi * self.f[idx] * (t - real_det_t))
        # calculate SNR
        SNR = self.calcSNR(self.f[idx] / 1000, 1, dist, format="active")
        # add noise
        noise = (10 ** (-SNR / 10)) * np.random.randn(t_length)
        recv_signal += noise
        A = np.column_stack(
            (np.sin(2 * np.pi * self.f[idx] * t), np.cos(2 * np.pi * self.f[idx] * t))
        )
        coeffs, _, _, _ = np.linalg.lstsq(A, recv_signal, rcond=None)
        sin_coeff, cos_coeff = coeffs
        phase_diff = np.arctan2(cos_coeff, sin_coeff)
        phase_diff = np.mod(phase_diff + np.pi, 2 * np.pi) - np.pi
        return phase_diff

    def time_estimate(self, signal, pulse):
        N = len(signal)
        M = len(pulse)
        J = np.zeros(N - M + 1)
        for n0 in range(N - M + 1):
            signal_dat = signal[n0 : n0 + M]
            J[n0] = np.dot(signal_dat, pulse)
        # signal_matrix = np.lib.stride_tricks.sliding_window_view(signal, M)
        # l = time.time()
        # Compute the dot product for each shifted version with the pulse
        # J = np.dot(signal_matrix, pulse)
        # Find the index of the maximum value in J
        n0hat = np.argmax(J)
        time_delay = n0hat / self.f0
        return time_delay

    # time delay using correlation
    def calc_timeDelay(self, real_t, f_idx=0):
        ret_delayt = np.zeros_like(real_t)
        K = 3
        t_origin = np.linspace(
            0, K * 2 * np.pi, int(K * self.f0 / self.f[f_idx]), dtype=np.float64
        )
        y_origin = np.sin(t_origin)
        T = 2 * K / self.f[f_idx]
        if real_t >= 1.0:
            raise NotImplementedError  # too large delay
        y_rec1 = np.zeros(int(self.f0 * T))
        y_rec2 = np.zeros(int(self.f0 * T))
        rt_idx = real_t * self.f0
        int_rt_idx = int(rt_idx)
        rt_idx -= int_rt_idx - 10
        y_rec1[int(rt_idx) : int(rt_idx) + int(K * self.f0 / self.f[f_idx])] = y_origin
        y_rec2[
            int(rt_idx) + 1 : int(rt_idx) + int(K * self.f0 / self.f[f_idx]) + 1
        ] = y_origin
        k_yrec2 = rt_idx - int(rt_idx)
        y_rec = k_yrec2 * y_rec2 + (1 - k_yrec2) * y_rec1

        SNR = self.calcSNR(
            self.f[f_idx] / 1000, 1, real_t * self.c / 2, format="active"
        )
        r_SNR = 10 ** (-SNR / 10)
        y_rec = y_rec + np.random.normal(0, r_SNR, size=y_rec2.shape)
        return self.time_estimate(y_rec, y_origin) + (int_rt_idx - 10) / self.f0

    def calcPosit(self, real_posit, idx=0):
        calc_posit = np.zeros(3)
        real_dposit = real_posit + self.hyd_posit
        real_delayt = np.linalg.norm(real_dposit, axis=1) / self.c * 2
        # calc delayt one by one
        calc_phaset = np.array(
            [self.get_phasedelay(real_delayt[i] * self.c / 2, idx) for i in range(4)]
        )

        if abs(calc_phaset[0] - calc_phaset[1]) > np.pi:
            calc_phaset[0] -= np.sign(calc_phaset[0] - calc_phaset[1]) * 2 * np.pi
        if abs(calc_phaset[3] - calc_phaset[2]) > np.pi:
            calc_phaset[3] -= np.sign(calc_phaset[3] - calc_phaset[2]) * 2 * np.pi
        dphasex = calc_phaset[1] - calc_phaset[0]
        dphasey = calc_phaset[3] - calc_phaset[2]
        # time calc
        calc_delayt = self.calc_timeDelay(real_delayt[0], f_idx=idx) / 2
        # calculate position
        calc_posit[0] = (
            (self.c)
            / (2 * np.pi * self.f[idx] * self.d)
            * (dphasex)
            * (self.c * calc_delayt)
        )
        calc_posit[1] = (
            (self.c)
            / (2 * np.pi * self.f[idx] * self.d)
            * (dphasey)
            * (self.c * calc_delayt)
        )
        calc_posit[2] = np.sqrt(
            (self.c * calc_delayt) ** 2 - calc_posit[0] ** 2 - calc_posit[1] ** 2
        )
        return calc_posit
    
