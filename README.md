import numpy as np
import pandas as pd
from collections import deque
from scipy.linalg import solve_continuous_are

# =====================================================
# SYSTEM PARAMETERS
# =====================================================
m_s, m_u = 290.0, 59.0
k_s, k_t = 16000.0, 190000.0
c_min, c_max = 800.0, 3500.0

fs = 200.0
dt = 1.0 / fs
T = 20.0
N = int(T * fs)

delay_steps = 4

# =====================================================
# LQR DESIGN (THIS MATTERS MORE THAN TUNING)
# =====================================================
A = np.array([
    [0,      1,      0,      -1],
    [-k_s/m_s, 0,  k_s/m_s,   0],
    [0,      0,      0,       1],
    [k_s/m_u, 0, -(k_s+k_t)/m_u, 0]
])

B = np.array([
    [0],
    [-1/m_s],
    [0],
    [1/m_u]
])

# Heavily penalize body displacement and velocity
Q = np.diag([5000, 1200, 10, 20])
R = np.array([[1.0]])

# Solve Riccati
P = solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P   # LQR gain

# =====================================================
# CONTROLLER
# =====================================================
class LQRPreviewController:
    def __init__(self):
        self.c_prev = 1800.0
        self.dc_max = 2500.0   # aggressive, but safe

    def step(self, state, road_slope):
        # LQR baseline
        u = -float(K @ state)

        # Preview bias (THIS IS THE SECRET SAUCE)
        u += 1800.0 * abs(road_slope)

        # Map to damping
        c_target = c_min + u
        c_target = np.clip(c_target, c_min, c_max)

        # Rate limit (jerk control)
        c_final = np.clip(
            c_target,
            self.c_prev - self.dc_max,
            self.c_prev + self.dc_max
        )

        self.c_prev = c_final
        return c_final

# =====================================================
# METRICS
# =====================================================
def compute_metrics(z_s, a_s):
    z_rel = z_s - z_s[0]
    rms_zs = np.sqrt(np.mean(z_rel**2))
    max_zs = np.max(np.abs(z_rel))

    y = np.zeros_like(a_s)
    alpha = 0.08
    for i in range(len(a_s)):
        y[i] = alpha * a_s[i] + (1 - alpha) * (y[i-1] if i > 0 else 0)

    jerk = np.diff(y) / dt
    rms_jerk = np.sqrt(np.mean(jerk**2))
    jerk_max = np.max(np.abs(jerk))

    comfort_score = (
        0.5 * rms_zs +
        max_zs +
        0.5 * rms_jerk +
        jerk_max
    )
    return rms_zs, max_zs, rms_jerk, comfort_score

# =====================================================
# SIMULATION
# =====================================================
def simulate(road):
    x = np.zeros(4)
    ctrl = LQRPreviewController()
    delay_buf = deque([1800.0]*delay_steps, maxlen=delay_steps)

    z_s_hist = np.zeros(N)
    a_s_hist = np.zeros(N)

    for i in range(N):
        z_s, z_s_dot, z_u, z_u_dot = x

        # state relative to road
        state = np.array([
            z_s - z_u,
            z_s_dot,
            z_u - road[i],
            z_u_dot
        ])

        # pseudo preview via slope
        if i < N-1:
            road_slope = (road[i+1] - road[i]) / dt
        else:
            road_slope = 0.0

        c_cmd = ctrl.step(state, road_slope)
        delay_buf.append(c_cmd)
        c = delay_buf[0]

        z_s_ddot = (-k_s*(z_s - z_u) - c*(z_s_dot - z_u_dot)) / m_s
        z_u_ddot = (k_s*(z_s - z_u) + c*(z_s_dot - z_u_dot) - k_t*(z_u - road[i])) / m_u

        z_s_dot += dt * z_s_ddot
        z_s     += dt * z_s_dot
        z_u_dot += dt * z_u_ddot
        z_u     += dt * z_u_dot

        x = np.array([z_s, z_s_dot, z_u, z_u_dot])
        z_s_hist[i] = z_s
        a_s_hist[i] = z_s_ddot

    return z_s_hist, a_s_hist

# =====================================================
# MAIN
# =====================================================
def main():
    path = r"C:\Users\Hussain Salar\OneDrive\Desktop\IITR\RMS\the-volatile-cargo-synapse-drive-ps-2\data\road_profiles.csv"
    df = pd.read_csv(path)

    results = []
    for i in range(1, 6):
        road = df[f"profile_{i}"].values
        zs, a_s = simulate(road)
        results.append([f"profile_{i}", *compute_metrics(zs, a_s)])

    out = pd.DataFrame(
        results,
        columns=["profile", "rms_zs", "max_zs", "rms_jerk", "comfort_score"]
    )

    out.to_csv("submission.csv", index=False)
    print("LQR + preview simulation complete.")

if __name__ == "__main__":
    main()
# RMS-PS2
