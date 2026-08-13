"""Demo kịch bản: mô phỏng 1 hành trình qua các tình huống DMS.

Chạy: python -m src.demo  (từ thư mục gốc repo)
Không cần camera/model — dùng mock backend, temporal logic chạy thật.
"""

import numpy as np

try:
    from .pipeline import SafetyAndHealthMonitorPipeline
    from .tier1 import MockLandmarkBackend
except ImportError:
    from pipeline import SafetyAndHealthMonitorPipeline
    from tier1 import MockLandmarkBackend

FPS = 10
FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


def run_scenario(monitor, backend, name, metrics, telematics, seconds, t0):
    print(f"\n--- {name} ---")
    backend.set_scenario(**metrics)
    alerts = []
    for i in range(int(seconds * FPS)):
        now = t0 + i / FPS
        out = monitor.process_stream_frame(FRAME, telematics, now=now)
        if out and (not alerts or alerts[-1] != out):
            print(f"  [t={now:6.1f}s] 🔊 {out}")
            alerts.append(out)
    if not alerts:
        print("  (không có cảnh báo — đúng kỳ vọng)" if "bình thường" in name
              else "  (không có cảnh báo)")
    return t0 + seconds


def main():
    # Demo kịch bản luôn dùng mock backend (metrics theo kịch bản);
    # chạy model thật: run_webcam.py / run_video.py / eval_fl3d.py
    backend = MockLandmarkBackend()
    monitor = SafetyAndHealthMonitorPipeline(edge_vlm_path="quantized_vlm.gguf",
                                             tier1_backend=backend)

    telem_normal = {"continuous_driving_min": 20, "speed_kmh": 45,
                    "ambient_temp_c": 28, "weather": "normal"}
    telem_long_hot = {"continuous_driving_min": 125, "speed_kmh": 52,
                      "ambient_temp_c": 35, "weather": "sunny_dusty"}

    t = 0.0
    t = run_scenario(monitor, backend, "Lái bình thường",
                     dict(ear=0.30), telem_normal, 3, t)
    t = run_scenario(monitor, backend, "Nhắm mắt kéo dài (T0 — khẩn cấp <300ms)",
                     dict(ear=0.10), telem_normal, 3, t)
    t = run_scenario(monitor, backend, "Nhìn điện thoại (T1 — khẩn cấp)",
                     dict(ear=0.30, phone_conf=0.9), telem_normal, 2, t)
    t = run_scenario(monitor, backend, "Ngáp liên tục (T3 — nhắc nhẹ + VLM)",
                     dict(ear=0.30, mar=0.75), telem_normal, 8, t)
    t = run_scenario(monitor, backend,
                     "Lái >2h trời nóng (T5 — VLM fusion telematics)",
                     dict(ear=0.30, mar=0.2), telem_long_hot, 2, t)

    print("\n--- Guardrails: chặn free text vi phạm y tế ---")
    for bad in ["Bạn có dấu hiệu thiếu máu, nên đi khám ngay.",
                "Huyet ap cua ban co ve thap.",
                "Chỉ số của bạn là 120/80."]:
        safe = monitor.enforce_medical_guardrails(bad, "T7_baseline_anomaly")
        print(f"  VLM (raw) : {bad}\n  Ra loa    : {safe}")


if __name__ == "__main__":
    main()
