
import math
import numpy as np
import threading
import cv2
import scipy.linalg
from ..shared.helpers import RunningVariance
from ..shared.settings import CFG

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]): return args[0]
        def decorator(func): return func
        return decorator

@njit(cache=True, fastmath=True)
def huber_weight(res_norm, delta=2.0):
    if res_norm <= delta:
        return 1.0
    return delta / res_norm

@njit(cache=True, fastmath=True)
def get_chi2_threshold_99(d):
    if d <= 0: return 0.0
    return d * (1.0 - 2.0 / (9.0 * d) + 2.32635 * math.sqrt(2.0 / (9.0 * d))) ** 3

@njit(cache=True, fastmath=True)
def get_chi2_threshold_95(d):
    if d <= 0: return 0.0
    return d * (1.0 - 2.0 / (9.0 * d) + 1.64485 * math.sqrt(2.0 / (9.0 * d))) ** 3

# Import EKF constants securely from CFG
STATIC_VAR_THR   = CFG["ekf_tuning"]["static_variance_threshold"]
STATIC_WIN       = CFG["ekf_tuning"]["static_variance_window"]
MIN_GRAV_SAMPLES = CFG["ekf_tuning"]["min_gravity_samples"]
MIN_FEAT_UPDATE  = CFG["ekf_tuning"]["min_feature_update"]
DEPTH_PATCH_R    = CFG["ekf_tuning"]["depth_patch_radius"]
DEPTH_MIN_MM     = CFG["ekf_tuning"]["depth_min_mm"]
DEPTH_MAX_MM     = CFG["ekf_tuning"]["depth_max_mm"]

# MSCKF Tracking parameters (fallback to defaults if not in CFG yet)
MSCKF_WINDOW     = CFG["ekf_tuning"].get("msckf_window_size", 12)
MIN_TRACK        = CFG["ekf_tuning"].get("min_feature_track_length", 4)

# --- IMU NOISE PARAMETERS ---
_BASE_ACCEL_ND  = 160e-6 * 9.81
_BASE_GYRO_ND   = np.deg2rad(0.007)

# Undertuned Bias Random Walk (BRW)
# Empirically tuned for underwater ROV (thermal gradients + vibration harmonics)
ACCEL_BRW = 2.0e-3 * 9.81         # 2.0 mg 
GYRO_BRW  = np.deg2rad(1.5 / 3600.0) # 0.00042 rad/s

VIB_MULTIPLIER = CFG["ekf_tuning"].get("imu_vibration_multiplier", 15.0)

ACCEL_ND = _BASE_ACCEL_ND * VIB_MULTIPLIER
GYRO_ND  = _BASE_GYRO_ND * VIB_MULTIPLIER

VIS_NOISE_P   = (0.010)**2
VIS_NOISE_PHI = (np.deg2rad(1.0))**2
REORTHO_INTERVAL = 500
VIS_NIS_CHI2_95 = 12.592
VIS_NIS_CHI2_99 = 16.812

# ============================================================
# NUMBA JIT KERNELS (Preserving your custom optimizations)
# ============================================================

@njit(cache=True, fastmath=True)
def skew(w):
    return np.array([
        [0.0, -w[2], w[1]],
        [w[2], 0.0, -w[0]],
        [-w[1], w[0], 0.0]
    ], dtype=np.float64)

@njit(cache=True)
def project_nullspace(H_f, r):
    """SVD extraction of the left null-space to eliminate 3D landmarks from the state."""
    U, S, Vt = np.linalg.svd(H_f, full_matrices=True)
    A = U[:, 3:] 
    r_o = A.T @ r
    return r_o, A

@njit(cache=True, fastmath=True)
def _rodrigues_jit(wx,wy,wz,out):
    t2=wx*wx+wy*wy+wz*wz; t=math.sqrt(t2)
    if t<1e-9:
        out[0,0]=1.0;out[0,1]=-wz;out[0,2]=wy;out[1,0]=wz;out[1,1]=1.0;out[1,2]=-wx;out[2,0]=-wy;out[2,1]=wx;out[2,2]=1.0;return
    
    if t > 3.13: 
        t = 3.13
        scale = 3.13 / math.sqrt(t2)
        wx *= scale; wy *= scale; wz *= scale
        
    c=math.cos(t);s=math.sin(t);tc=1.0-c;it=1.0/t;x=wx*it;y=wy*it;z=wz*it
    out[0,0]=tc*x*x+c;out[0,1]=tc*x*y-s*z;out[0,2]=tc*x*z+s*y;out[1,0]=tc*x*y+s*z;out[1,1]=tc*y*y+c;out[1,2]=tc*y*z-s*x;out[2,0]=tc*x*z-s*y;out[2,1]=tc*y*z+s*x;out[2,2]=tc*z*z+c

@njit(cache=True, fastmath=True)
def _mat3_mul(A,B,out):
    for i in range(3):
        for j in range(3):
            s=0.0
            for k in range(3): s+=A[i,k]*B[k,j]
            out[i,j]=s

@njit(cache=True, fastmath=True)
def _mat3_vec(A,v,out):
    out[0]=A[0,0]*v[0]+A[0,1]*v[1]+A[0,2]*v[2];out[1]=A[1,0]*v[0]+A[1,1]*v[1]+A[1,2]*v[2];out[2]=A[2,0]*v[0]+A[2,1]*v[1]+A[2,2]*v[2]

@njit(cache=True, fastmath=True)
def _triple_product_15(F,P,Qd,out,tmp):
    for i in range(15):
        for j in range(15):
            s=0.0
            for k in range(15): s+=F[i,k]*P[k,j]
            tmp[i,j]=s
    for i in range(15):
        for j in range(15):
            s=0.0
            for k in range(15): s+=tmp[i,k]*F[j,k]
            out[i,j]=s+Qd[i,j]

@njit(cache=True, fastmath=True)
def _symmetrise_15(P):
    for i in range(15):
        for j in range(i+1,15): avg=0.5*(P[i,j]+P[j,i]);P[i,j]=avg;P[j,i]=avg

@njit(cache=True, fastmath=True)
def _build_F_and_Qd_jit(F,Qd,R,a_b,w_b,dt,na_var,ng_var,nba_var,nbg_var):
    for i in range(15):
        for j in range(15): F[i,j]=0.0;Qd[i,j]=0.0
        F[i,i]=1.0
    F[0,3]=dt;F[1,4]=dt;F[2,5]=dt;ax,ay,az=a_b[0],a_b[1],a_b[2]
    
    for i in range(3):
        c0 = -R[i,1]*az + R[i,2]*ay
        c1 =  R[i,0]*az - R[i,2]*ax
        c2 = -R[i,0]*ay + R[i,1]*ax
        F[3+i, 6] = c0*dt
        F[3+i, 7] = c1*dt
        F[3+i, 8] = c2*dt

    wx,wy,wz=w_b[0],w_b[1],w_b[2]
    F[6,6]=1.0;F[6,7]=wz*dt;F[6,8]=-wy*dt;F[7,6]=-wz*dt;F[7,7]=1.0;F[7,8]=wx*dt;F[8,6]=wy*dt;F[8,7]=-wx*dt;F[8,8]=1.0
    F[6,12]=-dt;F[7,13]=-dt;F[8,14]=-dt
    Qd[3,3]=Qd[4,4]=Qd[5,5]=na_var*dt;Qd[6,6]=Qd[7,7]=Qd[8,8]=ng_var*dt;Qd[9,9]=Qd[10,10]=Qd[11,11]=nba_var*dt;Qd[12,12]=Qd[13,13]=Qd[14,14]=nbg_var*dt

@njit(cache=True, fastmath=True)
def _propagate_state_jit(p,v,R,ba,bg,accel_raw,gyro_raw,dt,gravity_world,F,Qd,P,dR,dR_half,R_mid,a_w_mid,a_b,w_b,w_dt,na_var,ng_var,nba_var,nbg_var,step_count,reortho_interval,tmp15):
    a_b[0]=accel_raw[0]-ba[0];a_b[1]=accel_raw[1]-ba[1];a_b[2]=accel_raw[2]-ba[2];w_b[0]=gyro_raw[0]-bg[0];w_b[1]=gyro_raw[1]-bg[1];w_b[2]=gyro_raw[2]-bg[2]
    w_dt[0]=w_b[0]*dt*0.5;w_dt[1]=w_b[1]*dt*0.5;w_dt[2]=w_b[2]*dt*0.5;_rodrigues_jit(w_dt[0],w_dt[1],w_dt[2],dR_half);_mat3_mul(R,dR_half,R_mid);_mat3_vec(R_mid,a_b,a_w_mid)
    a_w_mid[0]-=gravity_world[0];a_w_mid[1]-=gravity_world[1];a_w_mid[2]-=gravity_world[2];dt2h=0.5*dt*dt
    p[0]+=v[0]*dt+a_w_mid[0]*dt2h;p[1]+=v[1]*dt+a_w_mid[1]*dt2h;p[2]+=v[2]*dt+a_w_mid[2]*dt2h;v[0]+=a_w_mid[0]*dt;v[1]+=a_w_mid[1]*dt;v[2]+=a_w_mid[2]*dt
    w_dt[0]=w_b[0]*dt;w_dt[1]=w_b[1]*dt;w_dt[2]=w_b[2]*dt;_rodrigues_jit(w_dt[0],w_dt[1],w_dt[2],dR);_mat3_mul(R,dR,dR_half)
    for i in range(3):
        for j in range(3): R[i,j]=dR_half[i,j]
    step_count+=1
    if step_count%reortho_interval==0:
        n0=math.sqrt(R[0,0]**2+R[0,1]**2+R[0,2]**2)
        if n0>1e-12: R[0,0]/=n0;R[0,1]/=n0;R[0,2]/=n0
        d=R[1,0]*R[0,0]+R[1,1]*R[0,1]+R[1,2]*R[0,2];R[1,0]-=d*R[0,0];R[1,1]-=d*R[0,1];R[1,2]-=d*R[0,2]
        n1=math.sqrt(R[1,0]**2+R[1,1]**2+R[1,2]**2)
        if n1>1e-12: R[1,0]/=n1;R[1,1]/=n1;R[1,2]/=n1
        R[2,0]=R[0,1]*R[1,2]-R[0,2]*R[1,1];R[2,1]=R[0,2]*R[1,0]-R[0,0]*R[1,2];R[2,2]=R[0,0]*R[1,1]-R[0,1]*R[1,0]
    _build_F_and_Qd_jit(F,Qd,R,a_b,w_b,dt,na_var,ng_var,nba_var,nbg_var);_triple_product_15(F,P,Qd,P,tmp15);_symmetrise_15(P)
    return step_count

@njit(cache=True, fastmath=True)
def _batch_depth_lookup_jit(depth_map,xs,ys,r,h,w,min_mm,max_mm):
    n=len(xs);result=np.zeros(n,dtype=np.float64)
    for idx in range(n):
        xi=int(round(xs[idx]));yi=int(round(ys[idx]))
        if xi<r: xi=r
        if xi>=w-r: xi=w-r-1
        if yi<r: yi=r
        if yi>=h-r: yi=h-r-1
        vc=0;ps=(2*r+1)*(2*r+1);vals=np.empty(ps,dtype=np.float64)
        for dy in range(-r,r+1):
            for dx in range(-r,r+1):
                d=float(depth_map[yi+dy,xi+dx])
                if d>=min_mm and d<=max_mm: vals[vc]=d;vc+=1
        if vc>0:
            for i in range(vc):
                for j in range(i+1,vc):
                    if vals[j]<vals[i]: vals[i],vals[j]=vals[j],vals[i]
            result[idx]=vals[vc//2]
    return result

@njit(cache=True, fastmath=True)
def _mat_to_rotvec_jit(R):
    val=(R[0,0]+R[1,1]+R[2,2]-1.0)*0.5
    if val>1.0: val=1.0
    if val<-1.0: val=-1.0
    theta=math.acos(val);out=np.zeros(3)
    if theta<1e-9: return out
    k=theta/(2.0*math.sin(theta))
    out[0]=(R[2,1]-R[1,2])*k;out[1]=(R[0,2]-R[2,0])*k;out[2]=(R[1,0]-R[0,1])*k;return out

@njit(cache=True, fastmath=True)
def _compute_visual_jacobians_jit(n_obs, P_world, obs_curr, p, R_imu_T, R_ci, t_ci, fx, fy, cx, cy, z_m, depth_max_m, H_total, r_total, W_huber, R_obs_diag):
    for i in range(n_obs):
        p_w = P_world[i]
        p_curr_i = np.zeros(3)
        for j in range(3):
            p_curr_i[j] = R_imu_T[j,0]*(p_w[0]-p[0]) + R_imu_T[j,1]*(p_w[1]-p[1]) + R_imu_T[j,2]*(p_w[2]-p[2])
            
        p_c = np.zeros(3)
        for j in range(3):
            p_c[j] = R_ci[j,0]*p_curr_i[0] + R_ci[j,1]*p_curr_i[1] + R_ci[j,2]*p_curr_i[2] + t_ci[j]
            
        inv_z = 1.0 / max(p_c[2], 1e-3)
        u_hat = fx * p_c[0] * inv_z + cx
        v_hat = fy * p_c[1] * inv_z + cy
        
        r0 = obs_curr[i,0] - u_hat
        r1 = obs_curr[i,1] - v_hat
        r_total[2*i] = r0
        r_total[2*i+1] = r1
        
        res_norm = math.sqrt(r0*r0 + r1*r1)
        w = 1.0
        if res_norm > 2.0:
            w = 2.0 / res_norm
        W_huber[2*i] = w
        W_huber[2*i+1] = w
        
        J_proj = np.array([
            [fx*inv_z, 0.0, -fx*p_c[0]*inv_z*inv_z],
            [0.0, fy*inv_z, -fy*p_c[1]*inv_z*inv_z]
        ])
        
        J_ci = np.zeros((2,3))
        for r in range(2):
            for c in range(3):
                J_ci[r,c] = J_proj[r,0]*R_ci[0,c] + J_proj[r,1]*R_ci[1,c] + J_proj[r,2]*R_ci[2,c]
                
        # H_p = J_proj @ R_ci @ (-R_imu_T)
        for r in range(2):
            for c in range(3):
                H_total[2*i+r, c] = -(J_ci[r,0]*R_imu_T[0,c] + J_ci[r,1]*R_imu_T[1,c] + J_ci[r,2]*R_imu_T[2,c])
                
        # H_th = J_proj @ R_ci @ skew(p_curr_i)
        p_curr_i_skew = np.array([
            [0.0, -p_curr_i[2], p_curr_i[1]],
            [p_curr_i[2], 0.0, -p_curr_i[0]],
            [-p_curr_i[1], p_curr_i[0], 0.0]
        ])
        for r in range(2):
            for c in range(3):
                H_total[2*i+r, 6+c] = J_ci[r,0]*p_curr_i_skew[0,c] + J_ci[r,1]*p_curr_i_skew[1,c] + J_ci[r,2]*p_curr_i_skew[2,c]

        # Depth-weighted noise
        sigma_px = 1.5 + 0.5 * (z_m[i] / depth_max_m)
        variance = (sigma_px*sigma_px) / w
        R_obs_diag[2*i] = variance
        R_obs_diag[2*i+1] = variance

# ============================================================
# STATE ESTIMATOR (15-DOF EKF + MSCKF & Pre-Integration)
# ============================================================

class VIO_EKF:
    def __init__(self):
        self._lock = threading.Lock()
        
        # 15-DOF State
        self.p=np.zeros(3); self.v=np.zeros(3)
        self.R=np.eye(3); self.ba=np.zeros(3); self.bg=np.zeros(3)
        self.P=np.diag([1e-6]*3+[1e-4]*3+[1e-6]*3+[1e-4]*3+[1e-4]*3).astype(np.float64)
        
        self._R_vis=np.diag([VIS_NOISE_P]*3+[VIS_NOISE_PHI]*3).astype(np.float64)
        self._I15=np.eye(15); self._I3=np.eye(3)
        self._F=np.zeros((15,15)); self._Qd=np.zeros((15,15))
        self._dR=np.eye(3); self._dR_half=np.eye(3); self._R_mid=np.eye(3)
        self._a_w_mid=np.zeros(3); self._a_b=np.zeros(3)
        self._w_b=np.zeros(3); self._w_dt=np.zeros(3)
        self._tmp33=np.zeros((3,3)); self._P_tmp=np.zeros((15,15))
        self._tmp15x15 = np.empty((15, 15), dtype=np.float64)
        
        # Pre-allocated Jacobian buffers for visual update
        self._H_total_buf = np.zeros((600, 15), dtype=np.float64)
        self._r_total_buf = np.zeros(600, dtype=np.float64)
        self._W_huber_buf = np.ones(600, dtype=np.float64)
        self._R_obs_diag_buf = np.zeros(600, dtype=np.float64)
        
        self.gravity_world=None; self.gravity_ready=False
        self._var_tracker=RunningVariance(STATIC_WIN)
        self._still_accels=[]; self.last_imu_ts=None
        self._kf_p=np.zeros(3); self._kf_R=np.eye(3); self._kf_set=False
        self._step_count=0
        
        # Safety Nets
        self._starvation_ticks = 0 
        self._last_v_p = None
        self._last_v_R = None
        self.residual_log = [] 
        
        # --- MSCKF / Manifold Pre-Integration Buffers ---
        self.window = []
        self.tracks = {}
        self.pre_dp = np.zeros(3, dtype=np.float64)
        self.pre_dv = np.zeros(3, dtype=np.float64)
        self.pre_dR = np.eye(3, dtype=np.float64)
        self.pre_dt = 0.0
        self._last_vis_cam_pose = None
        self._last_vis_ts = None
        self._vis_noise_scale = 1.0
        self._vis_reject_streak = 0
        self._vis_accept_count = 0
        self._vis_reject_count = 0
        self._vis_nis_ema = 0.0
        self._vis_last_nis = 0.0

    def feed_imu(self, a, g, ts):
        with self._lock: 
            a_clipped = np.clip(a, -25.0, 25.0)
            g_clipped = np.clip(g, -5.0, 5.0)
            
            # Manifold Pre-Integration relative to last keyframe
            if self.last_imu_ts is not None and self.gravity_ready:
                dt = ts - self.last_imu_ts
                if 0 < dt < 0.1:
                    unbiased_g = g_clipped - self.bg
                    unbiased_a = a_clipped - self.ba
                    _rodrigues_jit(unbiased_g[0]*dt, unbiased_g[1]*dt, unbiased_g[2]*dt, self._tmp33)
                    
                    self.pre_dp += self.pre_dv * dt + 0.5 * (self.pre_dR @ unbiased_a) * dt**2
                    self.pre_dv += (self.pre_dR @ unbiased_a) * dt
                    
                    _mat3_mul(self.pre_dR, self._tmp33, self._dR)
                    self.pre_dR[:] = self._dR
                    self.pre_dt += dt

            # Global 15-DOF State Propagation
            self._propagate(a_clipped, g_clipped, ts)
            
            if np.isnan(self.p).any() or np.isnan(self.R).any():
                print("  [CRITICAL] NaN detected in IMU Propagation! Reverting state.")
                if self._last_v_p is not None:
                    self.p[:] = self._last_v_p
                    self.R[:] = self._last_v_R
                    self.v[:] = np.zeros(3)

    def _propagate(self, accel_raw, gyro_raw, ts):
        # Your custom gravity and 15-DOF integration preserved exactly.
        norm=math.sqrt(accel_raw[0]**2+accel_raw[1]**2+accel_raw[2]**2)
        self._var_tracker.push(norm)
        
        if not self.gravity_ready:
            is_s = self._var_tracker.is_full() and self._var_tracker.variance() < STATIC_VAR_THR
            if is_s:
                self._still_accels.append(accel_raw.copy())
                if len(self._still_accels) >= MIN_GRAV_SAMPLES:
                    samples = np.array(self._still_accels)
                    norms = np.linalg.norm(samples, axis=1)
                    median_norm = float(np.median(norms))
                    
                    inlier_mask = np.abs(norms - median_norm) < 0.05 * median_norm
                    inliers = samples[inlier_mask]
                    
                    if len(inliers) >= MIN_GRAV_SAMPLES // 2:
                        gb = np.mean(inliers, axis=0)
                        gm = float(np.linalg.norm(gb))
                        if 9.5 <= gm <= 10.5:
                            gu = gb / gm
                            zd = np.array([0., 0., -1.])
                            v = np.cross(gu, zd)
                            s = np.linalg.norm(v)
                            c = np.dot(gu, zd)
                            if s < 1e-8: 
                                Ra = np.eye(3) if c > 0 else np.diag([1., -1., -1.])
                            else:
                                vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                                Ra = np.eye(3) + vx + vx@vx * ((1. - c) / (s * s))
                            self.R[:] = Ra
                            self.gravity_world = np.array([0., 0., -gm])
                            self.gravity_ready = True
                            print(f"\n  [EKF] Gravity: ‖g‖={gm:.4f} m/s^2 ({len(inliers)}/{len(samples)} inliers)")
                        else:
                            print(f"  [WARN] Gravity magnitude {gm:.2f} m/s^2 unrealistic. Recalibrating...")
                            self._still_accels = []
                    else:
                        print(f"  [WARN] Too many gravity outliers. Recollecting...")
                        self._still_accels = []
            else:
                self._still_accels = []
            self.last_imu_ts = ts
            return
            
        if self.last_imu_ts is None: 
            self.last_imu_ts = ts
            return
            
        dt = ts - self.last_imu_ts
        self.last_imu_ts = ts
        
        if dt <= 0 or dt > 0.1: 
            return
            
        self._step_count = _propagate_state_jit(
            self.p, self.v, self.R, self.ba, self.bg, accel_raw, gyro_raw, dt,
            self.gravity_world, self._F, self._Qd, self.P,
            self._dR, self._dR_half, self._R_mid, self._a_w_mid,
            self._a_b, self._w_b, self._w_dt,
            ACCEL_ND**2, GYRO_ND**2, ACCEL_BRW**2, GYRO_BRW**2,
            self._step_count, REORTHO_INTERVAL, self._tmp15x15
        )
        
        # Zero Velocity Update (ZUPT): Gently decay velocity to zero if static
        if self.gravity_ready and self._var_tracker.is_full() and self._var_tracker.variance() < STATIC_VAR_THR:
            self.v *= 0.98

    def set_keyframe(self):
        with self._lock: 
            self._kf_p[:] = self.p
            self._kf_R[:] = self.R
            self._kf_set = True
        
    def get_angular_velocity(self):
        with self._lock: 
            return self._w_b.copy()

    def get_visual_health(self):
        with self._lock:
            if self._vis_reject_streak >= 10:
                mode = "VISION_DEGRADED"
            elif self._vis_noise_scale > 2.0:
                mode = "VISION_ADAPTIVE"
            else:
                mode = "NOMINAL"
            return {
                "mode": mode,
                "noise_scale": float(self._vis_noise_scale),
                "reject_streak": int(self._vis_reject_streak),
                "accept_count": int(self._vis_accept_count),
                "reject_count": int(self._vis_reject_count),
                "nis_ema": float(self._vis_nis_ema),
                "nis_last": float(self._vis_last_nis),
            }

    def update_visual(self, prev_pts, curr_pts, prev_depth, K, T_ic, T_ci, frame_ts=None):
        """
        COMMERCIAL GRADE: Tightly-Coupled Depth-Inertial EKF Update with Huber Loss.
        Bypasses PnP posing and updates IMU state directly from pixel residuals.
        """
        with self._lock:
            if not self.gravity_ready: return False, 0
            if prev_pts is None or curr_pts is None or prev_depth is None: return False, 0
            if len(prev_pts) < MIN_FEAT_UPDATE: return False, 0

            fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
            
            # --- 1. DEPTH LOOKUP & 3D REPROJECTION ---
            depth_h, depth_w = prev_depth.shape[:2]
            scale_x, scale_y = depth_w / (2.0*cx), depth_h / (2.0*cy)
            xs, ys = (prev_pts[:, 0] * scale_x), (prev_pts[:, 1] * scale_y)
            z_mm = _batch_depth_lookup_jit(prev_depth, xs, ys, DEPTH_PATCH_R, depth_h, depth_w, DEPTH_MIN_MM, DEPTH_MAX_MM)
            valid = z_mm > 0.0
            if int(np.count_nonzero(valid)) < MIN_FEAT_UPDATE: return False, 0

            # Reset visual reference pose on time gaps
            if self._last_vis_ts is not None and frame_ts is not None:
                dt = float(frame_ts - self._last_vis_ts)
                if dt <= 0.0 or dt > 0.5:
                    self._last_v_p = None
                    self._last_v_R = None

            if self._last_v_p is None:
                self._last_v_p = self.p.copy()
                self._last_v_R = self.R.copy()
                self._last_vis_ts = frame_ts
                return False, 0
                
            # Transform points into IMU frame (3D landmarks relative to PREVIOUS IMU)
            z = z_mm[valid] * 1e-3
            X_c = (prev_pts[valid,0] - cx) * z / fx
            Y_c = (prev_pts[valid,1] - cy) * z / fy
            P_c_prev = np.stack([X_c, Y_c, z], axis=1)
            P_i_prev = (T_ic[:3,:3] @ P_c_prev.T + T_ic[:3,3:4]).T
            
            # Map landmarks to GLOBAL WORLD frame using previous IMU pose
            P_world = (self._last_v_R @ P_i_prev.T).T + self._last_v_p
            
            obs_curr = curr_pts[valid] # 2D observations in current frame
            n_obs = min(len(obs_curr), 300)
            
            # --- 2. ITERATIVE TIGHTLY-COUPLED UPDATE ---
            R_ic = T_ic[:3,:3]; t_ic = T_ic[:3,3]
            R_ci = T_ci[:3,:3]; t_ci = T_ci[:3,3]
            
            H_total = self._H_total_buf[:n_obs * 2]
            r_total = self._r_total_buf[:n_obs * 2]
            W_huber = self._W_huber_buf[:n_obs * 2]
            R_obs_diag = self._R_obs_diag_buf[:n_obs * 2]
            
            H_total.fill(0.0)
            
            R_imu_T = self.R.T
            
            _compute_visual_jacobians_jit(
                n_obs, P_world, obs_curr, self.p, R_imu_T, R_ci, t_ci,
                fx, fy, cx, cy, z, float(DEPTH_MAX_MM) * 1e-3,
                H_total, r_total, W_huber, R_obs_diag
            )

            # Measurement Noise Covariance
            R_obs = np.diag(R_obs_diag)
            
            S = H_total @ self.P @ H_total.T + R_obs
            try:
                # Optimized solve using Cholesky decomposition: K = (S⁻¹ H P)ᵀ
                c_and_lower = scipy.linalg.cho_factor(S, lower=True)
                K_gain = scipy.linalg.cho_solve(c_and_lower, H_total @ self.P).T
                dx = K_gain @ r_total
                
                # Apply Correction
                self.p += dx[0:3]
                self.v += dx[3:6]
                _rodrigues_jit(dx[6], dx[7], dx[8], self._tmp33)
                _mat3_mul(self.R, self._tmp33, self._dR)
                self.R[:] = self._dR
                self.ba += dx[9:12]
                self.bg += dx[12:15]
                
                I = np.eye(15)
                KH = K_gain @ H_total
                self.P = (I - KH) @ self.P @ (I - KH).T + K_gain @ R_obs @ K_gain.T
                self.P = 0.5 * (self.P + self.P.T)
                
                # NIS Logging
                nis = float(r_total @ np.linalg.solve(S, r_total))
                self._vis_last_nis = nis
                self._vis_nis_ema = 0.9 * self._vis_nis_ema + 0.1 * nis
                
                # Adaptive visual noise + consistency gating using dynamic Chi-squared threshold
                chi2_threshold = get_chi2_threshold_99(n_obs * 2)
                if nis > chi2_threshold * max(1.0, self._vis_noise_scale):
                    self._vis_reject_streak += 1
                    self._vis_reject_count += 1
                    self._vis_noise_scale = min(12.0, self._vis_noise_scale * 1.25)
                    
                    # Update reference pose with EKF's best estimate (current propagated state) on rejection
                    self._last_v_p = self.p.copy()
                    self._last_v_R = self.R.copy()
                    self._last_vis_ts = frame_ts
                    
                    self.residual_log.append({
                        "tick": self._step_count,
                        "type": "visual_reject_nis",
                        "nis": nis,
                        "noise_scale": float(self._vis_noise_scale),
                        "inliers": n_obs,
                    })
                    return False, n_obs

                # Dynamically calculate the 95% threshold for adaptive scaling
                chi2_threshold_95 = get_chi2_threshold_95(n_obs * 2)
                if nis > chi2_threshold_95:
                    self._vis_noise_scale = min(12.0, self._vis_noise_scale * 1.10)
                else:
                    self._vis_noise_scale = max(1.0, self._vis_noise_scale * 0.985)

            except (np.linalg.LinAlgError, ValueError):
                # Update reference pose on math error as well
                self._last_v_p = self.p.copy()
                self._last_v_R = self.R.copy()
                self._last_vis_ts = frame_ts
                return False, n_obs

            if np.isnan(self.p).any() or np.isnan(self.R).any() or np.isnan(self.P).any():
                if self._last_v_p is not None:
                    self.p[:] = self._last_v_p
                    self.R[:] = self._last_v_R
                self.P[:] = np.eye(15, dtype=np.float64) * 1e-3
                return False, n_obs

            self._last_v_p = self.p.copy()
            self._last_v_R = self.R.copy()
            self._vis_reject_streak = 0
            self._vis_accept_count += 1
            
            R_wc_corr = self.R @ T_ic[:3, :3]
            p_wc_corr = self.p + self.R @ T_ic[:3, 3]
            self._last_vis_cam_pose = (R_wc_corr.copy(), p_wc_corr.copy())
            self._last_vis_ts = frame_ts
            
            self.residual_log.append({
                "tick": self._step_count,
                "type": "visual_tight",
                "inliers": n_obs,
                "tracks": n_obs,
                "rmse": float(nis),
                "rejected": False,
            })
            return True, n_obs

    def add_visual_tracks(self, frame_id, features_dict, K, T_ic):
        """
        Replaces solvePnPRansac with Tightly-Coupled MSCKF feature tracking.
        Appends to the sliding window, triangulates lost features, and executes Null-Space projection.
        """
        with self._lock:
            if not self.gravity_ready: return False, 0
            
            # --- 1. STATE AUGMENTATION (MSCKF Window) ---
            R_cam = self.R @ T_ic[:3, :3]
            p_cam = self.p + self.R @ T_ic[:3, 3]
            
            self.window.append({
                "id": frame_id,
                "p": p_cam.copy(),
                "R": R_cam.copy()
            })
            
            if len(self.window) > MSCKF_WINDOW:
                self.window.pop(0) 
                
            # Reset Manifold pre-integration
            self.pre_dp[:] = 0.0; self.pre_dv[:] = 0.0
            self.pre_dR[:] = np.eye(3); self.pre_dt = 0.0
            
            # --- 2. TRACK MANAGEMENT ---
            current_ids = set(features_dict.keys())
            for fid, (u, v) in features_dict.items():
                if fid not in self.tracks:
                    self.tracks[fid] = {"obs": []}
                self.tracks[fid]["obs"].append((frame_id, u, v))
                
            # --- 3. MSCKF NULLSPACE UPDATE ---
            lost_ids = [fid for fid in self.tracks.keys() if fid not in current_ids]
            
            total_r_o_norm = 0.0
            processed_tracks = 0
            
            r_msckf = []
            H_msckf = []
            
            # Compute T_ci for coordinate mapping
            T_ci = np.eye(4)
            R_ic = T_ic[:3, :3]
            t_ic = T_ic[:3, 3]
            T_ci[:3, :3] = R_ic.T
            T_ci[:3, 3] = -R_ic.T @ t_ic
            
            for fid in lost_ids:
                track = self.tracks[fid]
                if len(track["obs"]) >= MIN_TRACK:
                    res = self._process_msckf_feature(track, K, T_ic, T_ci)
                    if res is not None:
                        r_o, H_o = res
                        # Chi-squared consistency gate for the feature track
                        track_nis = float(r_o @ r_o) / (1.5**2)
                        track_threshold = get_chi2_threshold_99(len(r_o))
                        if track_nis <= track_threshold:
                            r_msckf.append(r_o)
                            H_msckf.append(H_o)
                            total_r_o_norm += np.linalg.norm(r_o)
                            processed_tracks += 1
                del self.tracks[fid]
                
            # Apply EKF Update using stacked projected residuals and Jacobians
            if processed_tracks > 0:
                r_total = np.concatenate(r_msckf)
                H_total = np.vstack(H_msckf)
                
                R_obs = np.eye(len(r_total)) * (1.5**2)
                S = H_total @ self.P @ H_total.T + R_obs
                
                try:
                    # Solve for Kalman Gain: K = (S⁻¹ H P)ᵀ
                    K_gain = np.linalg.solve(S, H_total @ self.P).T
                    dx = K_gain @ r_total
                    
                    # Apply Correction to EKF State
                    self.p += dx[0:3]
                    self.v += dx[3:6]
                    _rodrigues_jit(dx[6], dx[7], dx[8], self._tmp33)
                    _mat3_mul(self.R, self._tmp33, self._dR)
                    self.R[:] = self._dR
                    self.ba += dx[9:12]
                    self.bg += dx[12:15]
                    
                    I = np.eye(15)
                    KH = K_gain @ H_total
                    self.P = (I - KH) @ self.P @ (I - KH).T + K_gain @ R_obs @ K_gain.T
                    self.P = 0.5 * (self.P + self.P.T)
                except np.linalg.LinAlgError:
                    pass
                
            # --- 4. SAFETY NETS & LOGGING ---
            if processed_tracks > 0:
                self._starvation_ticks = 0
                self._last_v_p = self.p.copy()
                self._last_v_R = self.R.copy()
                self.residual_log.append({
                    "tick": self._step_count, 
                    "type": "msckf_nullspace",
                    "avg_error_norm": float(total_r_o_norm / processed_tracks)
                })
            else:
                self._starvation_ticks += 1
                
            if np.isnan(self.p).any() or np.isnan(self.R).any() or np.isnan(self.P).any():
                print("  [CRITICAL] NaN detected in Visual Update! Reverting state.")
                if self._last_v_p is not None:
                    self.p[:] = self._last_v_p
                    self.R[:] = self._last_v_R
                self.P[:] = np.eye(15) * 1e-3
                return False, processed_tracks

            return True, processed_tracks

    def _process_msckf_feature(self, track, K, T_ic, T_ci):
        obs = track["obs"]
        first_obs, last_obs = obs[0], obs[-1]
        
        pose1 = next((p for p in self.window if p["id"] == first_obs[0]), None)
        pose2 = next((p for p in self.window if p["id"] == last_obs[0]), None)
        
        if not pose1 or not pose2: return None
        
        # Build Projection Matrices
        P1 = K @ np.hstack((pose1["R"].T, -pose1["R"].T @ pose1["p"].reshape(3,1)))
        P2 = K @ np.hstack((pose2["R"].T, -pose2["R"].T @ pose2["p"].reshape(3,1)))
        
        pt1 = np.array([[first_obs[1]], [first_obs[2]]], dtype=np.float32)
        pt2 = np.array([[last_obs[1]], [last_obs[2]]], dtype=np.float32)
        
        # Triangulate historical 3D Point
        p4d = cv2.triangulatePoints(P1.astype(np.float32), P2.astype(np.float32), pt1, pt2)
        p3d_w = (p4d[:3] / (p4d[3] + 1e-6)).flatten()
        
        r_stack, Hf_stack, Hx_stack = [], [], []
        
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        R_ci = T_ci[:3,:3]; t_ci = T_ci[:3,3]
        R_ic = T_ic[:3,:3]; t_ic = T_ic[:3,3]
        
        for frame_id, u, v in obs:
            c_pose = next((p for p in self.window if p["id"] == frame_id), None)
            if not c_pose: continue
            
            p_c = c_pose["R"].T @ (p3d_w - c_pose["p"])
            pc_x, pc_y, z_depth = p_c[0], p_c[1], p_c[2]
            
            if z_depth < 0.1: continue # Reject points behind camera
            
            u_hat = fx * (pc_x / z_depth) + cx
            v_hat = fy * (pc_y / z_depth) + cy
            
            r_stack.append([u - u_hat, v - v_hat])
            
            # Jacobian w.r.t landmark position
            dz_dp = np.array([
                [1/z_depth, 0, -pc_x/(z_depth**2)],
                [0, 1/z_depth, -pc_y/(z_depth**2)]
            ])
            Hf = dz_dp @ c_pose["R"].T
            Hf_stack.append(Hf)
            
            # Jacobian w.r.t EKF state (recovered IMU pose)
            R_imu = c_pose["R"] @ R_ci
            p_imu = c_pose["p"] - R_imu @ t_ic
            p_curr_i = R_imu.T @ (p3d_w - p_imu)
            
            J_proj = np.array([[fx/z_depth, 0, -fx*pc_x/(z_depth**2)],
                               [0, fy/z_depth, -fy*pc_y/(z_depth**2)]])
            
            H_p = J_proj @ R_ci @ (-R_imu.T)
            H_th = J_proj @ R_ci @ skew(p_curr_i)
            
            Hx = np.zeros((2, 15), dtype=np.float64)
            Hx[:, :3] = H_p
            Hx[:, 6:9] = H_th
            Hx_stack.append(Hx)
            
        if len(r_stack) < MIN_TRACK: return None
        
        r_vec = np.vstack(r_stack).flatten()
        Hf_mat = np.vstack(Hf_stack)
        Hx_mat = np.vstack(Hx_stack)
        
        # Left Null-Space Projection (Hardware Accelerated)
        r_o, A = project_nullspace(Hf_mat, r_vec)
        H_o = A.T @ Hx_mat
        return r_o, H_o

    def get_pose(self):
        with self._lock:
            T = np.eye(4)
            T[:3,:3] = self.R.copy()
            T[:3,3]  = self.p.copy()
            
            idx = [0,1,2,6,7,8]
            c6 = self.P[np.ix_(idx,idx)].copy()
        return T, c6

    def is_ready(self):
        with self._lock: return self.gravity_ready