"""
TIGHTLY-COUPLED VIO — OAK-D S2 — KEYFRAME GATED & HW OPTIMIZED & HUD ENABLED
=============================================================================
"""

import depthai as dai
import cv2
import numpy as np
import os, json, time, threading, queue
from collections import deque
from datetime import timedelta
import sys, select, termios, tty
import math
import ctypes

# IMPORT OUR CUSTOM MODULES
from ..shared.settings import CFG
from ..shared.helpers import DepthDoubleBuffer, HAS_NUMBA, fast_underwater_restore
from ..estimation.state_estimator import VIO_EKF
from ..dashboard.server import HudServerContext, start_hud_server

# ============================================================
# CONFIG (DERIVED FROM JSON)
# ============================================================
SAVE_DIR   = CFG["paths"]["scan_dir"]
RGB_DIR    = os.path.join(SAVE_DIR, "rgb")
DEPTH_DIR  = os.path.join(SAVE_DIR, "depth")
POSES_FILE = os.path.join(SAVE_DIR, "poses.json")
LAYOUT_FILE = os.path.join(SAVE_DIR, "hud_layout.json")
os.makedirs(RGB_DIR, exist_ok=True)
os.makedirs(DEPTH_DIR, exist_ok=True)

TARGET_FPS = CFG["hardware"]["target_fps"]
IMU_RATE   = CFG["hardware"]["imu_rate_hz"]
USE_IR_PROJECTOR   = CFG["hardware"]["use_ir_projector"]
IR_DOT_BRIGHTNESS  = CFG["hardware"]["ir_dot_brightness_ma"]
LOCK_EXPOSURE      = CFG["hardware"]["lock_exposure"]
EXPOSURE_TIME_US   = CFG["hardware"]["exposure_time_us"]
ISO_SENSITIVITY    = CFG["hardware"]["iso_sensitivity"]
CALIBRATION_MODE   = CFG.get("hardware", {}).get("calibration_mode", "custom").lower()

GATE_MIN_FRAME_GAP   = CFG["keyframe_gating"]["min_frame_gap"]
GATE_MAX_FRAME_GAP   = CFG["keyframe_gating"]["max_frame_gap"]
GATE_MIN_DEPTH_VALID = CFG["keyframe_gating"]["min_depth_valid_ratio"]
GATE_MAX_BLUR_PIXELS = CFG["keyframe_gating"]["max_blur_pixels"]
LAPLACIAN_PASS_THRESHOLD = 50.0

QC_CFG = CFG.get("quality_control", {})
TARGET_SCORE_GOOD = QC_CFG.get("score_good_threshold", 0.75)
TARGET_DEPTH_PCT = QC_CFG.get("ideal_depth_ratio", 0.40) * 100.0
INFO_CFG = CFG.get("information_gating", {})
INFO_SCORE_MIN = INFO_CFG.get("min_information_score", 0.45)

STATIC_VAR_THR   = CFG["ekf_tuning"]["static_variance_threshold"]
MIN_GRAV_SAMPLES = CFG["ekf_tuning"]["min_gravity_samples"]
MIN_FEAT_UPDATE  = CFG["ekf_tuning"]["min_feature_update"]
DEPTH_MIN_MM     = CFG["ekf_tuning"]["depth_min_mm"]
DEPTH_MAX_MM     = CFG["ekf_tuning"]["depth_max_mm"]

BUDGET_CFG = CFG.get("runtime_budgets", {})
VIS_P95_BUDGET_MS = BUDGET_CFG.get("visual_p95_ms", 22.0)
DISK_P95_BUDGET_MS = BUDGET_CFG.get("disk_p95_ms", 14.0)
LOOP_P95_BUDGET_MS = BUDGET_CFG.get("main_loop_p95_ms", 35.0)
BUDGET_REPORT_SEC = BUDGET_CFG.get("report_interval_sec", 8.0)
QUEUE_PRESSURE_RAISE = BUDGET_CFG.get("queue_pressure_raise", 0.70)
QUEUE_PRESSURE_RECOVER = BUDGET_CFG.get("queue_pressure_recover", 0.35)

SYNC_CFG = CFG.get("time_sync", {})
SYNC_WARN_ABS_S = SYNC_CFG.get("warn_abs_offset_s", 0.030)
SYNC_WARN_DRIFT_SPS = SYNC_CFG.get("warn_drift_s_per_s", 0.004)
SYNC_EWMA_ALPHA = SYNC_CFG.get("offset_ewma_alpha", 0.08)

RECOVERY_CFG = CFG.get("recovery_modes", {})
VISION_WEAK_MIN_GAP = RECOVERY_CFG.get("vision_weak_min_frame_gap", 4)
IMU_ONLY_MIN_GAP = RECOVERY_CFG.get("imu_only_min_frame_gap", 8)
IMU_ONLY_HOLD_SEC = RECOVERY_CFG.get("imu_only_hold_sec", 6.0)
DISK_PRESSURE_MIN_GAP = RECOVERY_CFG.get("disk_pressure_min_frame_gap", 10)
DISK_PRESSURE_SAVE_STRIDE = max(1, int(RECOVERY_CFG.get("disk_pressure_save_stride", 2)))
DISK_PRESSURE_QFILL = RECOVERY_CFG.get("disk_pressure_queue_fill", 0.7)
VISION_WEAK_INFO_BONUS = RECOVERY_CFG.get("vision_weak_info_bonus", 0.08)

CR_CFG = CFG.get("color_restore", {})
CR_ENABLED = CR_CFG.get("enabled", False)
CR_R_MAX = CR_CFG.get("r_max_gain", 3.0)
CR_G_MAX = CR_CFG.get("g_max_gain", 1.2)
CR_HUD = CR_CFG.get("apply_to_hud", True)
CR_REC = CR_CFG.get("apply_to_recording", True)

ISP_WIDTH  = 960
ISP_HEIGHT = 540

# ============================================================
# SHARED STATE & BUFFERS
# ============================================================
stream_state = {"latest_jpeg": None}
recording_event = threading.Event()
visual_queue = queue.Queue(maxsize=1)
stop_event   = threading.Event()

lk_queue = queue.Queue(maxsize=1)
disk_health = {"consecutive_drops": 0}

cam_ctrl_lock = threading.Lock()
cam_state = {
    "wb": 4600,
    "exp": EXPOSURE_TIME_US,
    "iso": ISO_SENSITIVITY
}

hud_telemetry = {
    "state": "IDLE",
    "score": 1.0,
    "blur": 0.0,
    "depth_pct": 0.0,
    "message": "AWAITING GRAVITY CALIBRATION",
    "mode": "BOOT",
    "visual_nis": 0.0,
    "sync_offset_ms": 0.0,
    "sync_drift_ms_s": 0.0,
    "adaptive_gap": GATE_MIN_FRAME_GAP
}
hud_lock = threading.Lock()
runtime_state = {"bad_streak_counter": 0}
adaptive_gate = {"min_frame_gap": int(GATE_MIN_FRAME_GAP)}
imu_sync = {"last_imu_ts": None, "last_imu_host_ts": None}
sync_state = {"offset_ewma": None, "drift_ewma": 0.0, "last_cam_ts": None, "last_host_ts": None}
recovery_state = {"imu_only_until": 0.0}
lat_lock = threading.Lock()
latency_samples = {
    "visual_ms": deque(maxlen=600),
    "disk_ms": deque(maxlen=800),
    "main_loop_ms": deque(maxlen=600),
}
runtime_mode = {"mode": "BOOT"}

hud_thread = start_hud_server(HudServerContext(
    stream_state=stream_state,
    stop_event=stop_event,
    recording_event=recording_event,
    hud_telemetry=hud_telemetry,
    hud_lock=hud_lock,
    cam_ctrl_lock=cam_ctrl_lock,
    cam_state=cam_state,
    layout_file=LAYOUT_FILE,
    exposure_time_us=EXPOSURE_TIME_US,
    iso_sensitivity=ISO_SENSITIVITY,
    target_score_good=TARGET_SCORE_GOOD,
    laplacian_pass_threshold=LAPLACIAN_PASS_THRESHOLD,
    target_depth_pct=TARGET_DEPTH_PCT,
))

def _observe_latency(key, value_ms):
    with lat_lock:
        latency_samples[key].append(float(value_ms))

def _latency_percentiles(key):
    with lat_lock:
        arr = np.array(latency_samples[key], dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "count": int(arr.size),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }

def _feature_coverage_score(features, width, height):
    if not features:
        return 0.0
    xs = [f.position.x for f in features]
    ys = [f.position.y for f in features]
    if len(xs) < 4:
        return 0.0
    spread = (max(xs) - min(xs)) * (max(ys) - min(ys))
    return float(np.clip(spread / float(width * height), 0.0, 1.0))

def _parallax_score(curr_fd, prev_fd):
    if not curr_fd or not prev_fd:
        return 0.0
    shared = [fid for fid in curr_fd.keys() if fid in prev_fd]
    if len(shared) < 8:
        return 0.0
    disp = []
    for fid in shared:
        p0 = prev_fd[fid]
        p1 = curr_fd[fid]
        disp.append(math.hypot(p1[0] - p0[0], p1[1] - p0[1]))
    med = float(np.median(np.array(disp, dtype=np.float64)))
    return float(np.clip(med / 20.0, 0.0, 1.0))

# ============================================================
# ZERO-COPY BUFFERS
# ============================================================
MAX_FEATURES = 150
pp_buffer = np.empty((MAX_FEATURES, 2), dtype=np.float32)
pc_buffer = np.empty((MAX_FEATURES, 2), dtype=np.float32)

# ============================================================
# THREAD AFFINITY
# ============================================================
def pin_thread(core_id):
    """Pin calling thread to a specific core."""
    try:
        os.sched_setaffinity(0, {core_id})
    except AttributeError:
        pass  # non-Linux systems

# ============================================================
# THREADS
# ============================================================
def visual_worker(ekf, K_mat, T_ic, T_ci):
    pin_thread(1) # Core 1 dedicated to Visual EKF
    ok = 0
    fail = 0
    while not stop_event.is_set():
        try: 
            item = visual_queue.get(timeout=0.1)
        except queue.Empty: 
            continue
            
        if item is None: 
            break

        t0 = time.perf_counter()
        # Disentangle the tuple: old signature items (0-3) and dict items (4-5)
        if len(item) > 4 and isinstance(item[5], dict) and runtime_mode["mode"] == "MSCKF":
            # Tightly-Coupled MSCKF Path
            r, n = ekf.add_visual_tracks(item[4], item[5], K_mat, T_ic)
        else:
            # Tightly-Coupled Depth-Inertial EKF Update
            r, n = ekf.update_visual(item[0], item[1], item[2], K_mat, T_ic, T_ci, item[3])
        _observe_latency("visual_ms", (time.perf_counter() - t0) * 1000.0)
        if r: ok += 1
        else: fail += 1

        health = ekf.get_visual_health()
        with hud_lock:
            hud_telemetry["visual_nis"] = float(health.get("nis_ema", 0.0))
            hud_telemetry["mode"] = health.get("mode", runtime_mode["mode"])
        visual_queue.task_done()

def imu_worker(device, ekf):
    pin_thread(3)  # Core 3 dedicated to IMU/EKF
    q = device.getOutputQueue("imu", maxSize=100, blocking=False)
    ab = np.zeros(3, dtype=np.float64)
    gb = np.zeros(3, dtype=np.float64)
    last_err_ts = 0.0
    
    while not stop_event.is_set():
        try:
            m = q.tryGet() 
            if m is not None:
                for pkt in m.packets:
                    a = pkt.acceleroMeter
                    g = pkt.gyroscope
                    try:
                        ts = a.getTimestampDevice().total_seconds()
                    except AttributeError:
                        try: ts = a.timestamp.get().total_seconds()
                        except AttributeError: ts = time.monotonic()
                            
                    ab[0], ab[1], ab[2] = a.x, a.y, a.z
                    gb[0], gb[1], gb[2] = g.x, g.y, g.z
                    ekf.feed_imu(ab, gb, ts)
                    imu_sync["last_imu_ts"] = float(ts)
                    imu_sync["last_imu_host_ts"] = time.monotonic()
            else:
                time.sleep(0.001)
        except Exception as e:
            if stop_event.is_set(): break
            now = time.monotonic()
            if now - last_err_ts > 2.0:
                print(f"[WARN] IMU worker error: {e}")
                last_err_ts = now
            time.sleep(0.01)

def lk_worker():
    # Gutted LK Fallback Thread
    while not stop_event.is_set():
        try:
            lk_queue.get(timeout=0.1)
        except queue.Empty:
            continue

# ============================================================
# ASYNC DISK I/O WORKER
# ============================================================
disk_queue = queue.Queue(maxsize=200)

def disk_worker():
    pin_thread(2) # Core 2 dedicated to Disk I/O
    while not stop_event.is_set() or not disk_queue.empty():
        try:
            item = disk_queue.get(timeout=0.1)
            if item is None: break
            filepath, img = item[0], item[1]
            t0 = time.perf_counter()
            
            # Save raw Depth maps to .u16 Binary Array (Bypasses PNG compression CPU load)
            if filepath.endswith('.png'):
                filepath = filepath.replace('.png', '.u16')
                img.tofile(filepath)
            else:
                if len(item) == 3 and item[2]:
                    img = fast_underwater_restore(img, CR_R_MAX, CR_G_MAX)
                cv2.imwrite(filepath, img, [cv2.IMWRITE_JPEG_QUALITY, 75])
            _observe_latency("disk_ms", (time.perf_counter() - t0) * 1000.0)
            disk_queue.task_done()
        except queue.Empty: 
            continue

disk_thread = threading.Thread(target=disk_worker, daemon=False)
disk_thread.start()

# ============================================================
# PIPELINE CONFIG
# ============================================================
print("⏳ Waiting for OAK-D camera to be connected...")
while True:
    found, _ = dai.Device.getAnyAvailableDevice()
    if found:
        break
    time.sleep(1)
print("🔗 Camera detected! Booting hardware...")

pipeline = dai.Pipeline()

calib = None
cal_source = "UNKNOWN"

if CALIBRATION_MODE == "factory":
    try:
        with dai.Device() as temp_device:
            calib = temp_device.readFactoryCalibration()
            cal_source = "FACTORY"
            print("[CAL] ✓ Factory calibration loaded (as requested in config)")
    except Exception as e:
        print(f"[CAL] ⚠ Factory cal requested but not available: {e}. Falling back.")

if calib is not None:
    pipeline.setCalibrationData(calib)

cam = pipeline.create(dai.node.ColorCamera)
cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setIspScale(1, 2)
cam.setFps(TARGET_FPS)
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam.setPreviewSize(640, 360)
cam.setVideoSize(640, 360)

if LOCK_EXPOSURE:
    cam.initialControl.setManualExposure(cam_state["exp"], cam_state["iso"])
    cam.initialControl.setManualWhiteBalance(cam_state["wb"])
    print(f"[HW] Locked Camera Exposure: {cam_state['exp']/1000:.1f}ms / ISO {cam_state['iso']} / WB {cam_state['wb']}K")

controlIn = pipeline.create(dai.node.XLinkIn)
controlIn.setStreamName('control')
controlIn.out.link(cam.inputControl)

jpeg_enc = pipeline.create(dai.node.VideoEncoder)
jpeg_enc.setDefaultProfilePreset(TARGET_FPS, dai.VideoEncoderProperties.Profile.MJPEG)
jpeg_enc.setQuality(80)
cam.video.link(jpeg_enc.input)

xout_jpeg = pipeline.create(dai.node.XLinkOut)
xout_jpeg.setStreamName("mjpeg")
xout_jpeg.input.setBlocking(False)
xout_jpeg.input.setQueueSize(1)
jpeg_enc.bitstream.link(xout_jpeg.input)

mono_l = pipeline.create(dai.node.MonoCamera)
mono_r = pipeline.create(dai.node.MonoCamera)
mono_l.setBoardSocket(dai.CameraBoardSocket.CAM_B)
mono_r.setBoardSocket(dai.CameraBoardSocket.CAM_C)
mono_l.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_r.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_l.setFps(TARGET_FPS)
mono_r.setFps(TARGET_FPS)

stereo = pipeline.create(dai.node.StereoDepth)
if hasattr(dai.node.StereoDepth.PresetMode, "DEFAULT"):
    stereo_preset = dai.node.StereoDepth.PresetMode.DEFAULT
else:
    stereo_preset = dai.node.StereoDepth.PresetMode.HIGH_DENSITY
stereo.setDefaultProfilePreset(stereo_preset)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
stereo.setLeftRightCheck(True)

# Drop down Depth resolution directly to 640x360 to save vast USB bandwidth 
stereo.setOutputSize(640, 360) 
stereo.setSubpixel(True)
stereo.initialConfig.setConfidenceThreshold(200)

cfg = stereo.initialConfig.get()
cfg.postProcessing.spatialFilter.enable = True
cfg.postProcessing.spatialFilter.holeFillingRadius = 2
cfg.postProcessing.temporalFilter.enable = False
cfg.postProcessing.speckleFilter.enable = True
cfg.postProcessing.speckleFilter.speckleRange = 50
try:
    cfg.postProcessing.thresholdFilter.minRange = DEPTH_MIN_MM
    cfg.postProcessing.thresholdFilter.maxRange = DEPTH_MAX_MM
except Exception: 
    pass
stereo.initialConfig.set(cfg)
mono_l.out.link(stereo.left)
mono_r.out.link(stereo.right)

# Empower the VPU Feature Tracker to completely avoid the CPU LK fallback
feat = pipeline.create(dai.node.FeatureTracker)
feat.setHardwareResources(2, 2)  
feat_cfg = feat.initialConfig.get()
if hasattr(feat_cfg, "pyramidLevels"):
    feat_cfg.pyramidLevels = 5

if hasattr(feat_cfg, "cornerDetector") and hasattr(feat_cfg.cornerDetector, "cellGridDimension"):
    feat_cfg.cornerDetector.cellGridDimension = 4
feat.initialConfig.set(feat_cfg)
cam.isp.link(feat.inputImage)

imu_n = pipeline.create(dai.node.IMU)
imu_n.enableIMUSensor([dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW], reportRate=IMU_RATE)
imu_n.setBatchReportThreshold(5)
imu_n.setMaxBatchReports(20)

xout_i = pipeline.create(dai.node.XLinkOut)
xout_i.setStreamName("imu")
xout_i.input.setBlocking(False)
xout_i.input.setQueueSize(10)
imu_n.out.link(xout_i.input)

sync = pipeline.create(dai.node.Sync)
sync.setSyncThreshold(timedelta(milliseconds=30))
sync.inputs["rgb"].setBlocking(False)
sync.inputs["rgb"].setQueueSize(1)
sync.inputs["depth"].setBlocking(False)
sync.inputs["depth"].setQueueSize(1)
sync.inputs["features"].setBlocking(False)
sync.inputs["features"].setQueueSize(1)

cam.isp.link(sync.inputs["rgb"])
stereo.depth.link(sync.inputs["depth"])
feat.outputFeatures.link(sync.inputs["features"])

xout_s = pipeline.create(dai.node.XLinkOut)
xout_s.setStreamName("synced")
xout_s.input.setBlocking(False)
xout_s.input.setQueueSize(1)
sync.out.link(xout_s.input)

# ============================================================
# MAIN
# ============================================================
print(f"Starting VIO ({'Numba JIT' if HAS_NUMBA else 'numpy'})...")
print(f"HUD available at http://<PI_IP>:8080/ (client browser renders overlay)")

ekf = VIO_EKF()
saved_poses = []

with dai.Device(pipeline) as device:
    print("[SYS] Booting device and stabilizing pipeline (1.5s)...")
    time.sleep(1.5)

    if USE_IR_PROJECTOR:
        try:
            device.setIrLaserDotProjectorBrightness(IR_DOT_BRIGHTNESS)
            print(f"[HW] ✓ Active Stereo IR Projector ON ({IR_DOT_BRIGHTNESS}mA)")
        except Exception as e:
            print(f"[HW] ⚠ IR Projector NOT SUPPORTED on this model. (Running Passive Stereo)")

    if calib is None:
        try:
            calib = device.readCalibration()
            cal_source = "CUSTOM (EEPROM)"
            print("[CAL] ✓ Using current device calibration (custom EEPROM / fallback)")
        except Exception as e:
            print(f"[CAL] ⚠ Could not read any calibration: {e}")

    intr = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, ISP_WIDTH, ISP_HEIGHT)
    K_mat = np.array([
        [intr[0][0], 0,          intr[0][2]],
        [0,          intr[1][1], intr[1][2]],
        [0,          0,          1.0       ]], dtype=np.float64)

    with open(os.path.join(SAVE_DIR, "intrinsics.json"), "w") as f:
        json.dump({
            "width": ISP_WIDTH, "height": ISP_HEIGHT,
            "fx": intr[0][0], "fy": intr[1][1],
            "cx": intr[0][2], "cy": intr[1][2],
            "calibration_source": cal_source
        }, f, indent=2)

    try:
        ext = calib.getImuToCameraExtrinsics(dai.CameraBoardSocket.CAM_A, useSpecTranslation=True)
        T_ci = np.array(ext, dtype=np.float64)
        if T_ci.shape == (3,4): 
            T_ci = np.vstack([T_ci, [0,0,0,1]])
        T_ic = np.linalg.inv(T_ci)
    except Exception as e:
        T_ci = np.eye(4)
        T_ic = np.eye(4)

    imu_t = threading.Thread(target=imu_worker, args=(device,ekf), daemon=True)
    vis_t = threading.Thread(target=visual_worker, args=(ekf,K_mat, T_ic, T_ci), daemon=True)
    lk_thread = threading.Thread(target=lk_worker, daemon=True)
    
    imu_t.start()
    vis_t.start()
    lk_thread.start()

    qs = device.getOutputQueue("synced", 1, False)
    qj = device.getOutputQueue("mjpeg", 1, False) 
    control_q = device.getInputQueue("control")

    count = 0
    frame_ok = False
    prev_fd = {}
    depth_dbuf = None
    prev_depth = None  
    prev_ts = 0.0
    ekf_frame_counter = 0     
    last_saved_ekf_idx = 0
    
    current_applied_wb = 0
    current_applied_exp = 0
    current_applied_iso = 0

    gray_curr = None
    prev_gray = None

    print(f"\n[EKF] Hold still for gravity calibration...")

    has_tty = False
    old_settings = None
    try:
        if sys.stdin.isatty():
            old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            has_tty = True
        else:
            print("[WARN] No TTY detected.")
    except Exception as e:
        print(f"[WARN] Terminal setup failed: {e}. Key inputs ignored.")

    last_calib_print = 0
    last_heartbeat = time.time()
    
    telemetry_stats = {
        "visual_queue_drops": 0,
        "disk_queue_drops": 0,
        "forced_gaps_accepted": 0,
        "sync_warnings": 0,
        "budget_violations": 0,
        "mode_transitions": [],
        "recovered_from_pressure": 0
    }
    last_budget_report = time.monotonic()
    mode_last = "BOOT"

    try:
        while True:
            loop_t0 = time.perf_counter()
            with cam_ctrl_lock:
                target_wb = cam_state["wb"]
                target_exp = cam_state["exp"]
                target_iso = cam_state["iso"]
            
            if target_wb != current_applied_wb or target_exp != current_applied_exp or target_iso != current_applied_iso:
                ctrl = dai.CameraControl()
                ctrl.setManualWhiteBalance(target_wb)
                if LOCK_EXPOSURE:
                    ctrl.setManualExposure(target_exp, target_iso)
                control_q.send(ctrl)
                current_applied_wb = target_wb
                current_applied_exp = target_exp
                current_applied_iso = target_iso
                print(f"  [HW] Updated Settings -> WB: {target_wb}K | Exp: {target_exp}us | ISO: {target_iso}")

            if has_tty and select.select([sys.stdin], [], [], 0)[0]:
                k = sys.stdin.read(1).lower()
                if k == 'r':
                    if not ekf.is_ready(): 
                        print("  [WAIT] Gravity not calibrated yet — hold absolutely still!")
                    elif not recording_event.is_set():
                        recording_event.set()
                        runtime_state["bad_streak_counter"] = 0
                        print("\n[>>>] REC")
                        with hud_lock: hud_telemetry["state"] = "GOOD"
                elif k == 's' and recording_event.is_set():
                    recording_event.clear()
                    runtime_state["bad_streak_counter"] = 0
                    print(f"\n[|||] STOP {count}")
                    with hud_lock: hud_telemetry["state"] = "IDLE"; hud_telemetry["message"] = "PAUSED"
                elif k == 'q': 
                    break

            mj = qj.tryGet()
            if mj:
                # Lock-free atomic reference read from DepthAI
                stream_state["latest_jpeg"] = mj.getData().tobytes()

            sy = qs.tryGet()
            
            if sy or mj:
                last_heartbeat = time.time()
                
            if time.time() - last_heartbeat > 5.0:
                print("\n[FATAL] HARDWARE QUEUE STALL DETECTED! No frames for 5 seconds.")
                with hud_lock:
                    hud_telemetry["state"] = "BAD"
                    hud_telemetry["message"] = "FATAL: HARDWARE SENSOR STALL"
                break
            
            if sy:
                raw_rgb = sy["rgb"].getCvFrame()
                dep = sy["depth"].getFrame().astype(np.uint16)
                fm = sy["features"]
                
                try: rgb_ts = sy["rgb"].getTimestamp().total_seconds()
                except AttributeError: rgb_ts = time.monotonic()
                host_ts = time.monotonic()

                # Continuous camera/IMU time-offset and drift estimation.
                imu_ts = imu_sync["last_imu_ts"]
                if imu_ts is not None:
                    offset = float(rgb_ts - imu_ts)
                    prev_offset = sync_state["offset_ewma"]
                    if prev_offset is None:
                        sync_state["offset_ewma"] = offset
                    else:
                        sync_state["offset_ewma"] = (1.0 - SYNC_EWMA_ALPHA) * prev_offset + SYNC_EWMA_ALPHA * offset
                    if sync_state["last_cam_ts"] is not None and sync_state["last_host_ts"] is not None:
                        dt_host = max(1e-6, host_ts - sync_state["last_host_ts"])
                        drift = (offset - (prev_offset if prev_offset is not None else offset)) / dt_host
                        sync_state["drift_ewma"] = 0.9 * sync_state["drift_ewma"] + 0.1 * drift
                    sync_state["last_cam_ts"] = rgb_ts
                    sync_state["last_host_ts"] = host_ts

                    warn_sync = (
                        abs(sync_state["offset_ewma"]) > SYNC_WARN_ABS_S or
                        abs(sync_state["drift_ewma"]) > SYNC_WARN_DRIFT_SPS
                    )
                    if warn_sync:
                        telemetry_stats["sync_warnings"] += 1
                    with hud_lock:
                        hud_telemetry["sync_offset_ms"] = float(sync_state["offset_ewma"] * 1000.0)
                        hud_telemetry["sync_drift_ms_s"] = float(sync_state["drift_ewma"] * 1000.0)
                
                # Zero-Math Grayscale: Extracting the Green channel directly
                gray_curr = raw_rgb[:, :, 1].copy()
                
                rgb_to_save = raw_rgb
                apply_restore = bool(CR_ENABLED and CR_REC)
                
                if not frame_ok:
                    print(f"  [CAL] Frame: {raw_rgb.shape[1]}×{raw_rgb.shape[0]}")
                    frame_ok = True
                    
                if depth_dbuf is None:
                    depth_dbuf = DepthDoubleBuffer(dep.shape[0], dep.shape[1])
                depth_dbuf.write(dep)

                if count == 0 and not ekf.is_ready() and frame_ok:
                    now = time.time()
                    if now - last_calib_print > 3.0:
                        var = ekf._var_tracker.variance()
                        samples = len(ekf._still_accels)
                        print(f"  [EKF] Calibrating... var={var:.4f} (needs < {STATIC_VAR_THR}) "
                              f"samples={samples}/{MIN_GRAV_SAMPLES}")
                        last_calib_print = now
                        
                if ekf.is_ready() and not recording_event.is_set():
                    with hud_lock:
                        if hud_telemetry["message"] == "AWAITING GRAVITY CALIBRATION":
                            hud_telemetry["message"] = "READY - PRESS 'r' TO RECORD"

                if recording_event.is_set() and ekf.is_ready():
                    ekf_frame_counter += 1
                    gap = ekf_frame_counter - last_saved_ekf_idx
                    active_mode = runtime_mode["mode"]
                    
                    accept = False
                    reason = ""
                    valid_ratio = 0.0
                    evaluated_for_hud = False
                    is_forced_gap = False
                    
                    quality_score = 1.0       
                    quality_state = "GOOD"
                    info_score = 0.0
                    feature_cov_score = 0.0
                    parallax_score = 0.0
                    n_tracked = 0

                    if fm:
                        tracked = fm.trackedFeatures
                        n_tracked = len(tracked)
                        feature_cov_score = _feature_coverage_score(tracked, raw_rgb.shape[1], raw_rgb.shape[0])
                        cf_tmp = {t.id: (float(t.position.x), float(t.position.y)) for t in tracked}
                        parallax_score = _parallax_score(cf_tmp, prev_fd)
                    
                    # Optimized Temporal Gradient (Laplacian Variance) on center ROI to avoid downsampling aliasing
                    h_g, w_g = gray_curr.shape
                    roi = gray_curr[h_g//4 : 3*h_g//4, w_g//4 : 3*w_g//4]
                    laplacian_var = cv2.Laplacian(roi, cv2.CV_32F).var()

                    if gap >= GATE_MAX_FRAME_GAP:
                        accept = True
                        is_forced_gap = True
                        telemetry_stats["forced_gaps_accepted"] += 1
                        reason = f"forced_gap({gap})"
                        valid_ratio = float(np.count_nonzero(dep)) / dep.size
                        evaluated_for_hud = True
                        
                    elif gap >= adaptive_gate["min_frame_gap"]:
                        valid_ratio = float(np.count_nonzero(dep)) / dep.size
                        if valid_ratio < GATE_MIN_DEPTH_VALID:
                            reason = f"bad_depth({valid_ratio:.2f})"
                        else:
                            if laplacian_var < LAPLACIAN_PASS_THRESHOLD:
                                reason = f"blur_laplacian({laplacian_var:.1f})"
                            else:
                                # Information-aware gating: depth + blur + coverage + parallax.
                                qc_cfg = CFG.get("quality_control", {})
                                ideal_depth = qc_cfg.get("ideal_depth_ratio", 0.40)
                                severe_blur_equiv = qc_cfg.get("severe_blur_px", 10.0)
                                w_depth = INFO_CFG.get("weight_depth", 0.40)
                                w_blur = INFO_CFG.get("weight_blur", 0.25)
                                w_cov = INFO_CFG.get("weight_coverage", 0.20)
                                w_parallax = INFO_CFG.get("weight_parallax", 0.15)
                                q_depth = min(1.0, valid_ratio / ideal_depth) if ideal_depth > 0 else 0.0
                                q_blur = min(1.0, max(0.0, (laplacian_var - severe_blur_equiv) / 100.0))
                                info_score = (
                                    q_depth * w_depth +
                                    q_blur * w_blur +
                                    feature_cov_score * w_cov +
                                    parallax_score * w_parallax
                                )
                                info_thr = INFO_SCORE_MIN
                                if active_mode == "VISION_WEAK":
                                    info_thr = min(0.95, INFO_SCORE_MIN + VISION_WEAK_INFO_BONUS)
                                elif active_mode == "IMU_ONLY":
                                    info_thr = 1.0  # deterministic disable of visual-dependent acceptance

                                if info_score >= info_thr:
                                    accept = True
                                    reason = f"passed_info(I:{info_score:.2f},L:{laplacian_var:.1f},D:{valid_ratio:.2f})"
                                else:
                                    reason = f"low_info(I:{info_score:.2f},thr:{info_thr:.2f})"
                        evaluated_for_hud = True

                    if evaluated_for_hud:
                        qc_cfg = CFG.get("quality_control", {})
                        ideal_depth = qc_cfg.get("ideal_depth_ratio", 0.40)
                        severe_blur_equiv = 20.0
                        good_thresh = qc_cfg.get("score_good_threshold", 0.75)
                        weak_thresh = qc_cfg.get("score_weak_threshold", 0.40)
                        
                        w_depth = qc_cfg.get("weight_depth", 0.6) 
                        w_blur = qc_cfg.get("weight_blur", 0.4)
                        
                        q_depth = min(1.0, valid_ratio / ideal_depth) if ideal_depth > 0 else 0
                        q_blur  = min(1.0, max(0.0, (laplacian_var - severe_blur_equiv) / 100.0))
                        quality_score = (q_depth * w_depth) + (q_blur * w_blur)
                        if info_score > 0.0:
                            quality_score = 0.7 * quality_score + 0.3 * info_score
                        if is_forced_gap:
                            quality_state = "WEAK"
                            quality_score = min(quality_score, weak_thresh)
                        elif quality_score >= good_thresh:
                            quality_state = "GOOD"
                        elif quality_score >= weak_thresh:
                            quality_state = "WEAK"
                        else:
                            quality_state = "BAD"

                        if quality_state == "BAD":
                            runtime_state["bad_streak_counter"] += 1
                        else:
                            runtime_state["bad_streak_counter"] = 0

                        with hud_lock:
                            hud_telemetry["state"] = quality_state
                            hud_telemetry["score"] = quality_score
                            hud_telemetry["blur"] = laplacian_var 
                            hud_telemetry["depth_pct"] = valid_ratio
                            hud_telemetry["adaptive_gap"] = adaptive_gate["min_frame_gap"]
                            
                            if runtime_state["bad_streak_counter"] >= 8:
                                hud_telemetry["message"] = "CRITICAL: REVERSE TO LAST GOOD VIEW!"
                            elif quality_state == "BAD":
                                if laplacian_var < 35.0:
                                    hud_telemetry["message"] = "SLOW YAW / MOTION BLUR!"
                                elif valid_ratio < CFG.get("keyframe_gating", {}).get("min_depth_valid_ratio", 0.25):
                                    hud_telemetry["message"] = "MOVE CLOSER / POOR DEPTH!"
                                else:
                                    hud_telemetry["message"] = "TRACKING LOST!"
                            else:
                                hud_telemetry["message"] = ""

                    if accept:
                        if active_mode == "DISK_PRESSURE" and (ekf_frame_counter % DISK_PRESSURE_SAVE_STRIDE != 0):
                            accept = False
                            reason = f"disk_pressure_stride({DISK_PRESSURE_SAVE_STRIDE})"

                    if accept:
                        ekf.set_keyframe()
                        T, cov6 = ekf.get_pose()
                        Tc = T @ T_ic
                        
                        saved_poses.append({
                            "frame_id": count,
                            "ekf_frame_idx": ekf_frame_counter,
                            "gate_reason": reason,
                            "quality_score": float(quality_score),
                            "quality_state": quality_state,
                            "information_score": float(info_score),
                            "feature_coverage_score": float(feature_cov_score),
                            "parallax_score": float(parallax_score),
                            "tracked_features": int(n_tracked),
                            "sync_offset_ms": float(hud_telemetry.get("sync_offset_ms", 0.0)),
                            "recovery_mode": active_mode,
                            "is_forced_gap": is_forced_gap, 
                            "pose": Tc.tolist(),
                            "cov6": cov6.tolist()
                        })
                        
                        try:
                            disk_queue.put_nowait((os.path.join(RGB_DIR, f"{count:04d}.jpg"), rgb_to_save, apply_restore))
                            disk_queue.put_nowait((os.path.join(DEPTH_DIR, f"{count:04d}.png"), dep))
                            disk_health["consecutive_drops"] = 0
                        except queue.Full:
                            telemetry_stats["disk_queue_drops"] += 1
                            disk_health["consecutive_drops"] += 1
                            if disk_health["consecutive_drops"] >= 3:
                                with hud_lock:
                                    hud_telemetry["message"] = "CRITICAL: DISK SLOW! REDUCE SPEED!"
                                    hud_telemetry["state"] = "BAD"
                                print(f"  [CRITICAL] Disk queue saturated. Consider increasing GATE_MIN_FRAME_GAP.")
                            
                        count += 1
                        last_saved_ekf_idx = ekf_frame_counter
                        
                        if count % 10 == 0:
                            print(f"  [{count:04d} | idx:{ekf_frame_counter}] Health: {quality_score:.2f} ({quality_state}) | {reason}")

                if fm and gray_curr is not None:
                    cf = {t.id:(float(t.position.x),float(t.position.y)) for t in fm.trackedFeatures}
                    ds_curr = depth_dbuf.read() if depth_dbuf else None

                    if prev_fd and prev_depth is not None and ekf.is_ready():
                        if runtime_mode["mode"] != "IMU_ONLY":
                            pp, pc = [], []
                            for fid, c in cf.items():
                                if fid in prev_fd: 
                                    pp.append(prev_fd[fid])
                                    pc.append(c)
                                    
                            if len(pp) >= MIN_FEAT_UPDATE:
                                try: 
                                    n_pts = min(len(pp), MAX_FEATURES)
                                    pp_buffer[:n_pts] = pp[:n_pts]
                                    pc_buffer[:n_pts] = pc[:n_pts]
                                    visual_queue.put_nowait((
                                        pp_buffer[:n_pts].copy(), 
                                        pc_buffer[:n_pts].copy(), 
                                        prev_depth.copy(), 
                                        prev_ts, 
                                        count, 
                                        cf.copy()
                                    ))
                                except queue.Full: 
                                    telemetry_stats["visual_queue_drops"] += 1
                                    pass
                    
                    prev_fd = cf
                    prev_gray = gray_curr.copy()
                    # Use offset-compensated camera timestamp for visual update timing.
                    if sync_state["offset_ewma"] is not None:
                        prev_ts = rgb_ts - sync_state["offset_ewma"]
                    else:
                        prev_ts = rgb_ts
                    
                    if ds_curr is not None:
                        valid_ratio = float(np.count_nonzero(ds_curr)) / ds_curr.size
                        if valid_ratio > 0.15:
                            prev_depth = ds_curr.copy()

            time.sleep(0.001)

            _observe_latency("main_loop_ms", (time.perf_counter() - loop_t0) * 1000.0)
            now = time.monotonic()
            if now - last_budget_report >= BUDGET_REPORT_SEC:
                vstat = _latency_percentiles("visual_ms")
                dstat = _latency_percentiles("disk_ms")
                lstat = _latency_percentiles("main_loop_ms")
                vq_fill = visual_queue.qsize() / float(max(1, visual_queue.maxsize))
                dq_fill = disk_queue.qsize() / float(max(1, disk_queue.maxsize))
                pressure = max(vq_fill, dq_fill)
                budget_bad = (
                    vstat["p95"] > VIS_P95_BUDGET_MS or
                    dstat["p95"] > DISK_P95_BUDGET_MS or
                    lstat["p95"] > LOOP_P95_BUDGET_MS
                )
                health = ekf.get_visual_health()
                disk_pressure = (
                    dq_fill > DISK_PRESSURE_QFILL or
                    disk_health["consecutive_drops"] >= 2 or
                    dstat["p95"] > DISK_P95_BUDGET_MS
                )

                if health["mode"] == "VISION_DEGRADED":
                    recovery_state["imu_only_until"] = max(
                        recovery_state["imu_only_until"], now + IMU_ONLY_HOLD_SEC
                    )

                if disk_pressure:
                    mode = "DISK_PRESSURE"
                    adaptive_gate["min_frame_gap"] = max(adaptive_gate["min_frame_gap"], DISK_PRESSURE_MIN_GAP)
                elif now < recovery_state["imu_only_until"]:
                    mode = "IMU_ONLY"
                    adaptive_gate["min_frame_gap"] = max(adaptive_gate["min_frame_gap"], IMU_ONLY_MIN_GAP)
                elif health["mode"] == "VISION_ADAPTIVE" or vstat["p95"] > VIS_P95_BUDGET_MS or vq_fill > QUEUE_PRESSURE_RAISE:
                    mode = "VISION_WEAK"
                    adaptive_gate["min_frame_gap"] = max(adaptive_gate["min_frame_gap"], VISION_WEAK_MIN_GAP)
                else:
                    mode = "NOMINAL"
                    if pressure < QUEUE_PRESSURE_RECOVER and adaptive_gate["min_frame_gap"] > GATE_MIN_FRAME_GAP:
                        adaptive_gate["min_frame_gap"] -= 1
                        telemetry_stats["recovered_from_pressure"] += 1
                    elif pressure > QUEUE_PRESSURE_RAISE or budget_bad:
                        adaptive_gate["min_frame_gap"] = min(GATE_MAX_FRAME_GAP, adaptive_gate["min_frame_gap"] + 1)
                        telemetry_stats["budget_violations"] += 1

                runtime_mode["mode"] = mode
                if mode != mode_last:
                    telemetry_stats["mode_transitions"].append({
                        "t_monotonic": now,
                        "from": mode_last,
                        "to": mode,
                        "adaptive_gap": adaptive_gate["min_frame_gap"],
                        "visual_nis_ema": health.get("nis_ema", 0.0),
                        "disk_q_fill": dq_fill,
                        "visual_q_fill": vq_fill,
                    })
                    mode_last = mode

                with hud_lock:
                    hud_telemetry["mode"] = mode
                    if mode == "DISK_PRESSURE":
                        hud_telemetry["message"] = "DISK_PRESSURE: throttling saves"
                    elif mode == "IMU_ONLY":
                        hud_telemetry["message"] = "IMU_ONLY: visual updates paused"
                    elif mode == "VISION_WEAK":
                        hud_telemetry["message"] = "VISION_WEAK: tightening keyframe gate"

                print(
                    f"[BUDGET] mode={mode} gap={adaptive_gate['min_frame_gap']} "
                    f"V(p95={vstat['p95']:.1f}ms) D(p95={dstat['p95']:.1f}ms) "
                    f"L(p95={lstat['p95']:.1f}ms) q(v={vq_fill:.2f},d={dq_fill:.2f}) "
                    f"NIS={health.get('nis_ema', 0.0):.2f}"
                )
                last_budget_report = now

    except KeyboardInterrupt:
        print("\n[STOP]")
    finally:
        stop_event.set()
        recording_event.clear()
        
        while not visual_queue.empty():
            try: visual_queue.get_nowait()
            except: break
        visual_queue.put(None)
        
        while not disk_queue.empty():
            try: disk_queue.get_nowait()
            except: break
        disk_queue.put(None)

        imu_t.join(timeout=2.0)
        vis_t.join(timeout=2.0)
        lk_thread.join(timeout=2.0)
        hud_thread.join(timeout=2.0)
        disk_thread.join(timeout=30.0)
        
        if has_tty and old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            
        with open(POSES_FILE, "w") as f:
            json.dump(saved_poses, f, indent=2)
            
        res_file = os.path.join(SAVE_DIR, "vio_residuals.json")
        with open(res_file, "w") as f:
            json.dump(ekf.residual_log, f, indent=2)
            
        telem_file = os.path.join(SAVE_DIR, "session_telemetry.json")
        telemetry_stats["latency_summary_ms"] = {
            "visual": _latency_percentiles("visual_ms"),
            "disk": _latency_percentiles("disk_ms"),
            "main_loop": _latency_percentiles("main_loop_ms"),
        }
        telemetry_stats["final_adaptive_gap"] = adaptive_gate["min_frame_gap"]
        telemetry_stats["final_mode"] = runtime_mode["mode"]
        telemetry_stats["sync"] = {
            "offset_ms": float((sync_state["offset_ewma"] or 0.0) * 1000.0),
            "drift_ms_per_s": float(sync_state["drift_ewma"] * 1000.0),
        }
        telemetry_stats["ekf_visual_health"] = ekf.get_visual_health()
        with open(telem_file, "w") as f:
            json.dump(telemetry_stats, f, indent=2)
            
        print(f"\n[DONE] {count} keyframes saved → {POSES_FILE}")
        print(f"  VIO telemetry saved → {res_file}")
        print(f"  Session telemetry saved → {telem_file}")
        print(f"  Total EKF ticks monitored: {ekf_frame_counter}")
        if ekf_frame_counter > 0:
            print(f"  Retention rate: {(count/ekf_frame_counter)*100:.1f}%")