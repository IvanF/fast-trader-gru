"""Prometheus metrics for ML engine telemetry."""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable

from prometheus_client import Counter, Gauge, Histogram, Info, start_http_server

logger = logging.getLogger(__name__)

_health_check: Callable[[], dict[str, Any]] | None = None
_exit_plan_handler: Callable[[dict], dict[str, Any]] | None = None
_httpd: HTTPServer | None = None


class _HealthHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        if self.path not in ("/health", "/healthz"):
            self.send_response(404)
            self.end_headers()
            return

        if _health_check is None:
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "error", "reason": "health_check_not_registered"}).encode())
            return

        try:
            status = _health_check()
        except Exception as exc:
            status = {"status": "error", "reason": str(exc)}

        healthy = status.get("status") == "ok"
        code = 200 if healthy else 503
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(status).encode())

    def do_POST(self) -> None:
        if self.path != "/exit-plan":
            self.send_response(404)
            self.end_headers()
            return

        if _exit_plan_handler is None:
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "exit_plan_handler_not_registered"}).encode())
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            request = json.loads(body) if body else {}
            result = _exit_plan_handler(request)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
        except Exception as exc:
            logger.exception("exit-plan handler error")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(exc)}).encode())


def register_health_check(check: Callable[[], dict[str, Any]]) -> None:
    global _health_check
    _health_check = check


def register_exit_plan_handler(handler: Callable[[dict], dict[str, Any]]) -> None:
    global _exit_plan_handler
    _exit_plan_handler = handler


def start_health_server(port: int) -> None:
    global _httpd
    server = HTTPServer(("0.0.0.0", port), _HealthHandler)
    _httpd = server
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    logger.info("health server listening on :%d", port)

confidence_score = Gauge("ml_confidence_score", "Latest confidence score", ["symbol"])
volatility_multiplier = Gauge("ml_volatility_multiplier", "Latest volatility multiplier", ["symbol"])
regime_state = Gauge("ml_regime_state", "Market regime (0=Choppy,1=Trending,2=Breakout)", ["symbol"])
faiss_index_size = Gauge("ml_faiss_index_size", "FAISS memory index size")
tick_to_signal_latency = Histogram(
    "ml_tick_to_signal_latency_seconds",
    "Tick to signal latency",
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0],
)
redis_pubsub_latency = Histogram(
    "ml_redis_pubsub_latency_seconds",
    "Redis pub/sub processing latency",
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1],
)
signals_published = Counter("ml_signals_published_total", "Signals published to orders:signals")
gpu_vram_utilization = Gauge("ml_gpu_vram_utilization_percent", "GPU VRAM utilization")
cpu_utilization = Gauge("ml_cpu_utilization_percent", "CPU utilization")
btc_correlation = Gauge("ml_btc_correlation", "BTC correlation per symbol", ["symbol"])

# Gatekeeper
gatekeeper_accuracy = Gauge("ml_gatekeeper_accuracy", "Gatekeeper model accuracy (rolling)")
gatekeeper_sample_count = Gauge("ml_gatekeeper_sample_count", "Gatekeeper training samples")
gatekeeper_train_count = Gauge("ml_gatekeeper_train_count", "Gatekeeper retrain count")

# Model hot-swap
model_version = Gauge("ml_model_version_timestamp", "Loaded ONNX model manifest updated_at unix timestamp")
model_reloads = Counter("ml_model_hot_swap_total", "ONNX model hot-swap reload count (legacy)")
onnx_hot_swap_success = Counter("onnx_hot_swap_success_total", "Successful ONNX hot-swap reloads")
onnx_hot_swap_failures = Counter("onnx_hot_swap_failures_total", "Failed ONNX hot-swap reloads")
model_version_info = Info("ml_model_version", "Active ONNX model version ID")

# Rolling retrain
retrain_duration = Histogram(
    "ml_retrain_duration_seconds",
    "Rolling retrain subprocess duration",
    buckets=[30, 60, 120, 300, 600, 1200, 3600],
)
loss_delta = Gauge("ml_loss_delta", "Train loss improvement (final - initial, negative is better)")
last_training_timestamp = Gauge("last_training_timestamp", "Unix timestamp of last successful retrain")
retrain_running = Gauge("ml_retrain_running", "1 while retrain subprocess is active")
retrain_failures = Counter("ml_retrain_failures_total", "Failed retrain subprocess invocations")
retrain_trades_since_last = Gauge("ml_retrain_trades_since_last", "Closed trades since last retrain")

REGIME_MAP = {"Choppy": 0, "Trending": 1, "Breakout": 2}


def set_model_version(version: str, updated_at: float) -> None:
    model_version.set(updated_at)


def set_model_version_info(version: str, updated_at: float) -> None:
    model_version.set(updated_at)
    model_version_info.info({"version": version, "updated_at": str(int(updated_at))})


def record_hot_swap_success(version: str, updated_at: float) -> None:
    onnx_hot_swap_success.inc()
    model_reloads.inc()
    set_model_version_info(version, updated_at)


def record_hot_swap_failure() -> None:
    onnx_hot_swap_failures.inc()


def start_metrics_server(port: int) -> None:
    start_http_server(port)


def update_system_metrics() -> None:
    try:
        import psutil
        cpu_utilization.set(psutil.cpu_percent(interval=None))
    except Exception:
        pass
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_vram_utilization.set(100.0 * mem.used / mem.total)
        pynvml.nvmlShutdown()
    except Exception:
        gpu_vram_utilization.set(0.0)
