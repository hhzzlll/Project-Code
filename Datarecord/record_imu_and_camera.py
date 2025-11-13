"""
Unified controller: Start/stop Movella DOT IMU recording and OBS virtual camera together with hotkeys.

- Press 'r' to start BOTH: IMU onboard recording (after BT scan/connect/sync at 60 Hz) and OBS virtual camera
- Press 's' to stop BOTH: IMU onboard recording and OBS virtual camera, then export IMU CSVs to IMU_data/

Notes:
- IMU export is done over USB. After stopping recording, this script will keep polling for USB devices
  and export the most recent recording for each IMU that was started. It will show which devices
  are still missing and wait until all have been exported or timeout.
- Camera is controlled via OBS WebSocket (obsws-python) using Virtual Camera. If OBS isn't reachable, the IMU workflow still runs.

Dependencies:
- movelladot_pc_sdk (install the appropriate wheel from Python/x64)
- obsws-python (pip install obsws-python)
- Windows-only non-blocking keyboard input (msvcrt)
"""

# pyright: reportUnknownMemberType=false, reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportPrivateImportUsage=false

from __future__ import annotations

import os
import sys
import time
import msvcrt  # Windows-only non-blocking keyboard input
import threading
import ctypes
from datetime import datetime
from typing import Callable, Protocol, runtime_checkable, cast
from collections.abc import Iterable

import movelladot_pc_sdk
# local module import (works both when run as a script from repo root and from SensorCtrl)
try:
    import xdpchandler as xd  # type: ignore
except Exception:  # when running this file directly
    # ensure repo root is on sys.path, then import by package name
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    import xdpchandler as xd  # type: ignore

# Optional OBS imports
import importlib

try:
    obs_reqs = importlib.import_module("obsws_python.reqs")
except Exception:
    obs_reqs = None  # type: ignore


# ---------- Small helpers ----------

def _workspace_root() -> str:
    # SensorCtrl sits one level below repo root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _imu_data_dir() -> str:
    path = os.path.join(_workspace_root(), "Data\\imu_data\\IMU_data")
    os.makedirs(path, exist_ok=True)
    return path


def _wait_key(keys: set[str], prompt: str) -> str:
    print(prompt)
    while True:
        if msvcrt.kbhit():
            ch = msvcrt.getch()
            try:
                c = ch.decode("utf-8").lower()
            except Exception:
                c = ""
            if c in keys:
                print(f"\nKey '{c}' pressed.")
                return c
        time.sleep(0.05)


# ---------- OBS control (optional) ----------

@runtime_checkable
class ObsWsClient(Protocol):
    def get_version(self) -> object: ...
    def start_virtual_cam(self) -> object: ...
    def stop_virtual_cam(self) -> object: ...


class ObsController:
    def __init__(self, host: str = "localhost", port: int = 4455, password: str = "XmSt8BQ3Z0jSSP1Z") -> None:
        from typing import Any  # local import to avoid global Any when obsws not installed
        self._client: Any | None = None
        self._host: str = host
        self._port: int = port
        self._password: str = password

    def connect(self) -> bool:
        if not obs_reqs:
            print("[OBS] obsws-python 未安装，跳过相机控制。")
            return False
        try:
            # Compat across versions: ReqClient / ObsClient
            client_ctor: Callable[..., object]
            if hasattr(obs_reqs, "ReqClient"):
                client_ctor = cast(Callable[..., object], getattr(obs_reqs, "ReqClient"))
            elif hasattr(obs_reqs, "ObsClient"):
                client_ctor = cast(Callable[..., object], getattr(obs_reqs, "ObsClient"))
            else:
                print("[OBS] 未找到 obsws-python 客户端类 (ReqClient/ObsClient)")
                return False
            self._client = client_ctor(host=self._host, port=self._port, password=self._password)
            # quick probe
            assert self._client is not None
            getattr(self._client, "get_version")()
            print("[OBS] 已连接 OBS WebSocket")
            return True
        except Exception as e:
            print(f"[OBS] 连接失败: {e}")
            self._client = None
            return False

    def start_virtual_cam(self) -> None:
        if not self._client:
            return
        try:
            getattr(self._client, "start_virtual_cam")()
            print("[OBS] 启动虚拟摄像头")
        except Exception as e:
            print(f"[OBS] 启动虚拟摄像头失败: {e}")

    def stop_virtual_cam(self) -> None:
        if not self._client:
            return
        try:
            getattr(self._client, "stop_virtual_cam")()
            print("[OBS] 关闭虚拟摄像头")
        except Exception as e:
            print(f"[OBS] 关闭虚拟摄像头失败: {e}")


# ---------- IMU helpers ----------

def ensure_sync(manager: object, device_list: list[object]) -> bool:  # type: ignore[reportUnknownParameterType]
    if not device_list:
        return False
    root = device_list[-1].bluetoothAddress()
    print(f"\nStarting sync... Root node: {root}")
    print("This takes at least ~14 seconds on first attempt")
    if manager.startSync(root):
        return True
    print(f"Could not start sync. Reason: {manager.lastResultText()}")
    _SYNC_ERR = getattr(movelladot_pc_sdk, "XRV_SYNC_COULD_NOT_START", None)
    if _SYNC_ERR is None or manager.lastResult() != _SYNC_ERR:
        return False
    manager.stopSync()
    print("Retrying sync after stopping existing sync state...")
    return manager.startSync(root)


def configure_devices_for_onboard_recording(devices: Iterable[object]) -> None:
    for device in devices:
        if device.setOnboardFilterProfile("General"):
            print(f"{device.deviceTagName()}: Profile set to General")
        else:
            print(f"{device.deviceTagName()}: Failed to set profile. {device.lastResultText()}")
        print(f"{device.deviceTagName()}: Set onboard recording data rate to 60 Hz")
        if device.setOutputRate(60):
            print("Successfully set onboard recording rate")
        else:
            print(f"Setting onboard recording rate failed! {device.lastResultText()}")


def start_onboard_recording(devices: Iterable[object]) -> tuple[set[str], dict[str, int]]:
    """Start onboard recording on all devices.
    Returns (started_ids, started_index_by_id).
    """
    started_ids: set[str] = set()
    index_by_id: dict[str, int] = {}
    for device in devices:
        dev_id = device.deviceId().toXsString()
        print(f"Starting onboard recording for {device.deviceTagName()} ({dev_id}) ...")
        started = False
        if hasattr(device, "startRecording"):
            started = device.startRecording()
        else:
            started = device.startTimedRecording(2 * 60 * 60)
        if started:
            started_ids.add(dev_id)
            # reading current count as the expected index on export
            try:
                index_by_id[dev_id] = device.recordingCount()
            except Exception:
                pass
            print("OK")
        else:
            print(f"Failed: {device.lastResultText()}")
    return started_ids, index_by_id


def stop_onboard_recording(devices: Iterable[object]) -> None:
    for device in devices:
        print(f"Stopping onboard recording for {device.deviceTagName()} ...")
        stopped = False
        if hasattr(device, "stopRecording"):
            stopped = device.stopRecording()
        else:
            if hasattr(device, "stopTimedRecording"):
                try:
                    stopped = device.stopTimedRecording()
                except Exception:
                    stopped = False
        if stopped:
            print("OK")
        else:
            print(f"Failed or not supported: {device.lastResultText()}")


class SyncLogger:
    """线程安全的同步日志记录器：记录目标对齐时间与各项实际触发时间。"""
    def __init__(self) -> None:
        self._lock = threading.Lock()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path = os.path.join(_imu_data_dir(), f"sync_log_{ts}.csv")
        # 打开文件句柄并写入表头
        self._fh = open(self._path, "a", encoding="utf-8")
        self._fh.write("time_local,phase,role,device_id,target_mono,actual_mono,delta_ms,extra\n")
        self._fh.flush()

    def log(self, *, phase: str, role: str, device_id: str, target_mono: float, actual_mono: float, extra: str = "") -> None:
        delta_ms = (actual_mono - target_mono) * 1000.0
        time_local = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        line = f"{time_local},{phase},{role},{device_id},{target_mono:.6f},{actual_mono:.6f},{delta_ms:.3f},{extra}\n"
        with self._lock:
            self._fh.write(line)
            self._fh.flush()

    def log_target(self, *, phase: str, target_mono: float) -> None:
        # 仅记录目标时刻参考（无实际触发）
        time_local = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        line = f"{time_local},{phase},target,NA,{target_mono:.6f},,,\n"
        with self._lock:
            self._fh.write(line)
            self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


def _wait_until(target_monotonic: float) -> None:
    # 精细等待到目标单调时钟时间（避免系统时钟跳变）
    while True:
        now = time.perf_counter()
        remain = target_monotonic - now
        if remain <= 0:
            return
        # 自适应睡眠：>10ms 睡长一点，<10ms 短睡，<1ms 自旋
        if remain > 0.01:
            time.sleep(0.005)
        elif remain > 0.001:
            time.sleep(0.0005)
        else:
            # 亚毫秒，自旋
            pass


def coordinated_start(obs: ObsController, obs_ok: bool, devices: list[object], logger: SyncLogger | None = None) -> tuple[set[str], dict[str, int]]:
    """
    在同一时刻（下一秒边界）并行启动：OBS 虚拟摄像头 + 所有 IMU 板载录制。
    返回 (started_ids, index_by_id)。
    """
    go = threading.Event()
    target = time.perf_counter() + 1.0  # 下一秒附近
    if logger:
        logger.log_target(phase="start", target_mono=target)

    started_ids: set[str] = set()
    index_by_id: dict[str, int] = {}
    lock = threading.Lock()

    threads: list[threading.Thread] = []

    # OBS 线程
    if obs_ok:
        def _obs_job() -> None:
            go.wait()
            try:
                actual = time.perf_counter()
                obs.start_virtual_cam()
                if logger:
                    logger.log(phase="start", role="obs", device_id="obs", target_mono=target, actual_mono=actual)
            except Exception:
                pass
        t = threading.Thread(target=_obs_job, daemon=True)
        threads.append(t)

    # IMU 线程：每个设备一条，减少串行延迟
    for dev in devices:
        def _imu_job(d=dev) -> None:  # 绑定默认参数避免闭包引用变化
            go.wait()
            try:
                ok = False
                if hasattr(d, "startRecording"):
                    actual = time.perf_counter()
                    ok = d.startRecording()
                else:
                    actual = time.perf_counter()
                    ok = d.startTimedRecording(2 * 60 * 60)
                if ok:
                    with lock:
                        did = d.deviceId().toXsString()
                        started_ids.add(did)
                        try:
                            index_by_id[did] = d.recordingCount()
                        except Exception:
                            pass
                        # 记录日志：对齐/实际及设备 startUTC（若可用）
                        extra = ""
                        try:
                            ri = d.getRecordingTime()
                            extra = f"startUTC={ri.startUTC()}"
                        except Exception:
                            pass
                        if logger:
                            logger.log(phase="start", role="imu", device_id=did, target_mono=target, actual_mono=actual, extra=extra)
            except Exception:
                pass

        t = threading.Thread(target=_imu_job, daemon=True)
        threads.append(t)

    # 启动所有线程
    for t in threads:
        t.start()

    # 等到对齐时刻再广播开始
    _wait_until(target)
    now_mono = time.perf_counter()
    delta_ms = (now_mono - target) * 1000.0
    print(f"[SYNC] 已到达开始对齐时刻，正在同时触发 OBS 与 IMU (目标={target:.6f}, 实际={now_mono:.6f}, 偏差={delta_ms:.3f} ms)")
    if logger:
        logger.log(phase="start", role="host", device_id="host", target_mono=target, actual_mono=now_mono, extra="anchor")
    go.set()

    # 等待线程结束（给一点时间即可）
    for t in threads:
        t.join(timeout=2.0)

    # 日志
    if started_ids:
        print(f"并行启动完成，IMU 成功数量: {len(started_ids)} / {len(devices)}")
    else:
        print("并行启动未成功启动任何 IMU。")

    return started_ids, index_by_id


def coordinated_stop(obs: ObsController, obs_ok: bool, devices: list[object], logger: SyncLogger | None = None) -> None:
    """
    在同一时刻（下一秒边界）并行停止：OBS 虚拟摄像头 + 所有 IMU 录制。
    """
    go = threading.Event()
    target = time.perf_counter() + 1.0
    if logger:
        logger.log_target(phase="stop", target_mono=target)

    threads: list[threading.Thread] = []

    if obs_ok:
        def _obs_stop() -> None:
            go.wait()
            try:
                actual = time.perf_counter()
                obs.stop_virtual_cam()
                if logger:
                    logger.log(phase="stop", role="obs", device_id="obs", target_mono=target, actual_mono=actual)
            except Exception:
                pass
        threads.append(threading.Thread(target=_obs_stop, daemon=True))

    for dev in devices:
        def _imu_stop(d=dev) -> None:
            go.wait()
            try:
                if hasattr(d, "stopRecording"):
                    actual = time.perf_counter()
                    _ = d.stopRecording()
                    did = d.deviceId().toXsString()
                    if logger:
                        logger.log(phase="stop", role="imu", device_id=did, target_mono=target, actual_mono=actual)
                elif hasattr(d, "stopTimedRecording"):
                    try:
                        actual = time.perf_counter()
                        _ = d.stopTimedRecording()
                        did = d.deviceId().toXsString()
                        if logger:
                            logger.log(phase="stop", role="imu", device_id=did, target_mono=target, actual_mono=actual)
                    except Exception:
                        pass
            except Exception:
                pass
        threads.append(threading.Thread(target=_imu_stop, daemon=True))

    for t in threads:
        t.start()

    _wait_until(target)
    now_mono = time.perf_counter()
    delta_ms = (now_mono - target) * 1000.0
    print(f"[SYNC] 已到达停止对齐时刻，正在同时触发停止 (目标={target:.6f}, 实际={now_mono:.6f}, 偏差={delta_ms:.3f} ms)")
    if logger:
        logger.log(phase="stop", role="host", device_id="host", target_mono=target, actual_mono=now_mono, extra="anchor")
    go.set()

    for t in threads:
        t.join(timeout=2.0)


def _boost_timer_resolution() -> None:
    """Windows: 提升系统计时器分辨率到 1ms，降低调度抖动。"""
    try:
        ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception:
        pass


def _restore_timer_resolution() -> None:
    try:
        ctypes.windll.winmm.timeEndPeriod(1)
    except Exception:
        pass

def _reset_export_flags(xdpc_handler: xd.XdpcHandler) -> None:
    try:
        xdpc_handler._XdpcHandler__exportDone = False  # type: ignore[attr-defined]
        xdpc_handler._XdpcHandler__packetsReceived = 0  # type: ignore[attr-defined]
    except Exception:
        pass


def _select_standard_export_fields() -> object:
    exportData = movelladot_pc_sdk.XsIntArray()
    exportData.push_back(movelladot_pc_sdk.RecordingData_Timestamp)
    # exportData.push_back(movelladot_pc_sdk.RecordingData_Euler)
    # exportData.push_back(movelladot_pc_sdk.RecordingData_Acceleration)
    exportData.push_back(movelladot_pc_sdk.RecordingData_Quaternion)
    exportData.push_back(movelladot_pc_sdk.RecordingData_AngularVelocity)
    # exportData.push_back(movelladot_pc_sdk.RecordingData_MagneticField)
    exportData.push_back(movelladot_pc_sdk.RecordingData_Status)
    return exportData


def export_all_expected_over_usb(
    xdpc_handler: xd.XdpcHandler,
    expected_ids: set[str],
    expected_index_hint: dict[str, int],
    timeout_s: int = 1800,
) -> None:
    """Export recording for each expected device ID.
    - Keeps polling for USB devices until all are exported or timeout.
    - Exports to IMU_data/ device_<tag>_<index>.csv
    """
    if not expected_ids:
        print("[Export] 没有需要导出的设备。")
        return

    exported: set[str] = set()
    t0 = time.time()
    hint_printed = False

    print(f"[Export] 目标设备数量: {len(expected_ids)} -> {sorted(expected_ids)}")

    while time.time() - t0 <= timeout_s and len(exported) < len(expected_ids):
        remaining = sorted(list(expected_ids - exported))
        if not hint_printed:
            print("[Export] 请将 IMU 逐个通过 USB 连接到电脑，我会自动导出。")
            hint_printed = True
        print(f"[Export] 待导出设备剩余: {remaining}")

        xdpc_handler.detectUsbDevices()
        if len(xdpc_handler.detectedDots()) == 0:
            time.sleep(2.0)
            continue

        xdpc_handler.connectDots()
        usb_devices = list(xdpc_handler.connectedUsbDots())
        if len(usb_devices) == 0:
            time.sleep(2.0)
            continue

        for device in usb_devices:
            dev_id = device.deviceId().toXsString()
            if dev_id not in expected_ids or dev_id in exported:
                continue

            # Determine which recording index to export
            try:
                recordingIndex = device.recordingCount()
            except Exception:
                recordingIndex = expected_index_hint.get(dev_id, 0)

            recInfo = device.getRecordingInfo(recordingIndex)
            if recInfo.empty():
                print(f"[{dev_id}] 无法获取录制信息: {device.lastResultText()}")
                continue

            size_b = recInfo.storageSize()
            total_s = recInfo.totalRecordingTime()
            print(f"[{dev_id}] 录制索引[{recordingIndex}] 大小: {size_b} bytes, 时长: {total_s} s")

            export_fields = _select_standard_export_fields()
            if not device.selectExportData(export_fields):
                print(f"[{dev_id}] 选择导出字段失败: {device.lastResultText()}")
                continue

            csv_name = f"device_{device.deviceTagName()}_{recordingIndex}.csv"
            csv_path = os.path.join(_imu_data_dir(), csv_name)
            print(f"[{dev_id}] 导出到 {csv_path}")

            if not device.enableLogging(csv_path):
                print(f"[{dev_id}] 打开日志文件失败: {device.lastResultText()}")
                continue

            _reset_export_flags(xdpc_handler)
            if not device.startExportRecording(recordingIndex):
                print(f"[{dev_id}] 启动导出失败: {device.lastResultText()}")
                device.disableLogging()
                continue

            start_ms = movelladot_pc_sdk.XsTimeStamp_nowMs()
            while not xdpc_handler.exportDone() and movelladot_pc_sdk.XsTimeStamp_nowMs() - start_ms <= 10 * 60 * 1000:
                time.sleep(0.1)

            if xdpc_handler.exportDone():
                print(f"[{dev_id}] 导出完成。数据包: {xdpc_handler.packetsReceived()}")
                exported.add(dev_id)
            else:
                print(f"[{dev_id}] 导出超时，尝试停止导出...")
                if not device.stopExportRecording():
                    print(f"[{dev_id}] 停止导出失败: {device.lastResultText()}")
                else:
                    print(f"[{dev_id}] 已停止导出。")

            device.disableLogging()

        # brief pause before next scan
        time.sleep(1.0)

    # Final check
    if exported == expected_ids:
        print("[Export] 全部设备导出完成。")
    else:
        missing = sorted(list(expected_ids - exported))
        print(f"[Export] 超时仍未导出的设备: {missing}")


# ---------- Main workflow ----------

def main() -> None:
    # Prepare OBS controller (won't block if missing)
    _boost_timer_resolution()
    obs = ObsController()
    obs_ok = obs.connect()

    xdpc = xd.XdpcHandler()
    if not xdpc.initialize():
        xdpc.cleanup()
        sys.exit(-1)

    # 1) BT scan
    xdpc.scanForDots()
    if len(xdpc.detectedDots()) == 0:
        print("未发现 Movella DOT 设备，退出。")
        xdpc.cleanup()
        sys.exit(-1)

    # 2) Connect BT
    xdpc.connectDots()
    devices = xdpc.connectedDots()
    if len(devices) == 0:
        print("无法连接到任何 Movella DOT 设备，退出。")
        xdpc.cleanup()
        sys.exit(-1)

    # 3) Configure 60 Hz
    configure_devices_for_onboard_recording(devices)

    # 4) Sync
    manager = xdpc.manager()
    if not ensure_sync(manager, devices):
        print("同步启动失败，退出。")
        xdpc.cleanup()
        sys.exit(-1)

    # 5) Wait for 'r' to start both
    _wait_key({"r"}, "按 'r' 同时开始：IMU 录制 + OBS 虚拟摄像头...")

    # 并行对齐启动，尽量缩小 IMU 与摄像头以及各 IMU 间的启动时间差
    sync_logger = SyncLogger()
    started_ids, started_index_hint = coordinated_start(obs, obs_ok, devices, logger=sync_logger)
    if not started_ids:
        print("没有任何设备开始录制，退出。")
        manager.stopSync()
        xdpc.cleanup()
        sys.exit(-1)

    print("录制中... 按 's' 同时停止 IMU 与 OBS 虚拟摄像头。")

    # Non-blocking wait for 's' while printing time info
    start_utc_hint: int | None = None
    while True:
        # print timed info for the first device
        if devices:
            try:
                rec_info = devices[0].getRecordingTime()
                ts = movelladot_pc_sdk.XsTimeStamp()
                ts.setMsTime(rec_info.startUTC() * 1000)
                now_utc = int(time.time())
                dev_start_utc = rec_info.startUTC()
                if dev_start_utc > 0:
                    elapsed = max(0, now_utc - int(dev_start_utc))
                else:
                    if start_utc_hint is None:
                        start_utc_hint = now_utc
                        elapsed = 0
                    else:
                        elapsed = now_utc - start_utc_hint
                total = rec_info.totalRecordingTime()
                remaining = rec_info.remainingRecordingTime()
                unknown_remaining = (total >= 65535) or (remaining >= 65535)
                base = f"Recording start: {ts.utcToLocalTime().toXsString()} | elapsed: {elapsed} s"
                if not unknown_remaining:
                    base += f" | remaining: {remaining} s"
                print(f"{base}\r", end="", flush=True)
            except Exception:
                pass
        if msvcrt.kbhit():
            ch = msvcrt.getch()
            try:
                c = ch.decode("utf-8").lower()
            except Exception:
                c = ""
            if c == "s":
                print("\n's' detected. Stopping both...")
                break
        time.sleep(1.0)

    # 6) Stop IMU and OBS（并行对齐停止）
    coordinated_stop(obs, obs_ok, devices, logger=sync_logger)

    # Wait for callback that recording stopped
    t0 = time.time()
    while not xdpc.recordingStopped() and time.time() - t0 < 30:
        time.sleep(0.1)

    # Stop sync and close BT
    print("Stopping sync...")
    if not manager.stopSync():
        print("停止同步失败。")

    print("Closing BT ports...")
    manager.close()

    # Stop OBS recording
    if obs_ok:
        obs.stop_virtual_cam()

    # 7) USB export for all started devices
    print("\n开始导出 IMU 数据到 IMU_data 文件夹...")
    export_all_expected_over_usb(xdpc, started_ids, started_index_hint, timeout_s=3600)

    # Cleanup
    xdpc.cleanup()
    try:
        sync_logger.close()
    except Exception:
        pass
    _restore_timer_resolution()


if __name__ == "__main__":
    main()
