"""Background NVML sampler and the 20-parameter ``GPU_Metrics`` table.

Each parameter's ``Method`` column says how it was obtained:

* **Measured** — sampled from NVML (``nvidia-ml-py``) every second during the
  translation phase; average and peak are reported.
* **Derived** — computed from measured values; the formula is in ``Notes``.
* **Spec lookup** — peak datasheet figure from :mod:`gpu_specs`.
* **N/A** — not obtainable on this runtime, with the reason.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from shloka_translator.monitoring.gpu_specs import GPUSpec, lookup_gpu_spec
from shloka_translator.utils.timing import percentile

logger = logging.getLogger(__name__)

PCIE_GBPS_PER_LANE = {1: 0.25, 2: 0.5, 3: 0.985, 4: 1.969, 5: 3.938, 6: 7.563}
OPENAI_NOTE = (
    "Translation inference ran on OpenAI servers; local GPU figures reflect only local work "
    "(language detection, helpers)."
)


@dataclass
class GPUStatic:
    """Static device facts read once from NVML.

    Attributes:
        name: GPU name.
        total_mem_gb: Total VRAM (GB).
        max_graphics_clock: Max graphics clock (MHz).
        max_mem_clock: Max memory clock (MHz).
        bus_width: Memory bus width (bits) or ``None``.
        num_cores: CUDA cores from NVML or ``None``.
        pcie_gen_max: Max PCIe generation.
        pcie_width_max: Max PCIe lanes.
        pcie_gen_cur: Current PCIe generation.
        pcie_width_cur: Current PCIe lanes.
        driver: Driver version.
    """

    name: str = ""
    total_mem_gb: float = 0.0
    max_graphics_clock: float | None = None
    max_mem_clock: float | None = None
    bus_width: int | None = None
    num_cores: int | None = None
    pcie_gen_max: int | None = None
    pcie_width_max: int | None = None
    pcie_gen_cur: int | None = None
    pcie_width_cur: int | None = None
    driver: str = ""


@dataclass
class Samples:
    """Time series collected by the sampler.

    Attributes:
        t: Timestamps (monotonic seconds).
        util: GPU utilisation %.
        mem_util: Memory-controller utilisation %.
        mem_used_gb: VRAM used (GB).
        gfx_clock: Graphics clock MHz.
        mem_clock: Memory clock MHz.
        power_w: Power draw W.
        temp_c: Temperature °C.
        pcie_tx_gbs: PCIe TX GB/s.
        pcie_rx_gbs: PCIe RX GB/s.
        sm_util: Per-process SM utilisation % (if exposed).
    """

    t: list[float] = field(default_factory=list)
    util: list[float] = field(default_factory=list)
    mem_util: list[float] = field(default_factory=list)
    mem_used_gb: list[float] = field(default_factory=list)
    gfx_clock: list[float] = field(default_factory=list)
    mem_clock: list[float] = field(default_factory=list)
    power_w: list[float] = field(default_factory=list)
    temp_c: list[float] = field(default_factory=list)
    pcie_tx_gbs: list[float] = field(default_factory=list)
    pcie_rx_gbs: list[float] = field(default_factory=list)
    sm_util: list[float] = field(default_factory=list)


@dataclass
class RunStats:
    """Workload statistics needed for throughput/latency/efficiency rows.

    Attributes:
        records_processed: Records processed live (not cached/skipped).
        output_tokens: Output tokens of the selected model.
        total_tokens: Input + output tokens.
        latencies: Per-record latencies (s).
        duration_s: Translation-phase wall time (s).
        batch_size_desc: Text for the Batch Size row.
        provider: Provider of the selected model.
    """

    records_processed: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latencies: list[float] = field(default_factory=list)
    duration_s: float = 0.0
    batch_size_desc: str = ""
    provider: str = ""


def _avg(values: list[float]) -> float | None:
    """Mean of a list.

    Args:
        values: Numbers.

    Returns:
        Mean or ``None`` if empty.
    """
    return sum(values) / len(values) if values else None


def _fmt(value: float | None, digits: int = 2) -> str:
    """Format a number or ``N/A``.

    Args:
        value: Number.
        digits: Decimals.

    Returns:
        Formatted string.
    """
    return "N/A" if value is None else f"{value:,.{digits}f}"


class GPUMonitor:
    """Background thread sampling NVML once per interval.

    Example:
        >>> mon = GPUMonitor()
        >>> mon.start(); mon.stop()  # no-op without a GPU
    """

    def __init__(self, interval_s: float = 1.0, device_index: int = 0) -> None:
        """Create the monitor.

        Args:
            interval_s: Sampling period in seconds.
            device_index: GPU index.
        """
        self.interval_s = interval_s
        self.device_index = device_index
        self.samples = Samples()
        self.static = GPUStatic()
        self.available = False
        self.error = ""
        self._nvml: Any = None
        self._handle: Any = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._pcie_ok = True
        self._proc_ok = True
        self._last_proc_ts = 0
        self._init_nvml()

    def _init_nvml(self) -> None:
        """Initialise NVML and read static facts (sets ``available``)."""
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        except Exception as exc:  # noqa: BLE001
            self.error = f"NVML unavailable: {type(exc).__name__}"
            return
        self.available = True
        n, h, s = self._nvml, self._handle, self.static

        def safe(fn: Any, *args: Any) -> Any:
            """Call an NVML function, returning ``None`` on error.

            Args:
                fn: NVML function.
                *args: Arguments.

            Returns:
                Result or ``None``.
            """
            try:
                return fn(*args)
            except Exception:  # noqa: BLE001
                return None

        name = safe(n.nvmlDeviceGetName, h) or ""
        s.name = name.decode() if isinstance(name, bytes) else str(name)
        mem = safe(n.nvmlDeviceGetMemoryInfo, h)
        s.total_mem_gb = mem.total / 1024**3 if mem else 0.0
        s.max_graphics_clock = safe(n.nvmlDeviceGetMaxClockInfo, h, n.NVML_CLOCK_GRAPHICS)
        s.max_mem_clock = safe(n.nvmlDeviceGetMaxClockInfo, h, n.NVML_CLOCK_MEM)
        s.bus_width = safe(getattr(n, "nvmlDeviceGetMemoryBusWidth", lambda _h: None), h)
        s.num_cores = safe(getattr(n, "nvmlDeviceGetNumGpuCores", lambda _h: None), h)
        s.pcie_gen_max = safe(n.nvmlDeviceGetMaxPcieLinkGeneration, h)
        s.pcie_width_max = safe(n.nvmlDeviceGetMaxPcieLinkWidth, h)
        s.pcie_gen_cur = safe(n.nvmlDeviceGetCurrPcieLinkGeneration, h)
        s.pcie_width_cur = safe(n.nvmlDeviceGetCurrPcieLinkWidth, h)
        drv = safe(n.nvmlSystemGetDriverVersion) or ""
        s.driver = drv.decode() if isinstance(drv, bytes) else str(drv)

    def _sample(self) -> None:
        """Take one sample of every dynamic metric."""
        n, h, s = self._nvml, self._handle, self.samples
        s.t.append(time.monotonic())
        try:
            u = n.nvmlDeviceGetUtilizationRates(h)
            s.util.append(float(u.gpu))
            s.mem_util.append(float(u.memory))
        except Exception:
            logger.debug("utilisation sample failed", exc_info=True)
        for fn, target, scale in (
            (lambda: n.nvmlDeviceGetMemoryInfo(h).used, s.mem_used_gb, 1 / 1024**3),
            (lambda: n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_GRAPHICS), s.gfx_clock, 1),
            (lambda: n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_MEM), s.mem_clock, 1),
            (lambda: n.nvmlDeviceGetPowerUsage(h), s.power_w, 1 / 1000),
            (lambda: n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU), s.temp_c, 1),
        ):
            try:
                target.append(float(fn()) * scale)
            except Exception:
                logger.debug("sample failed", exc_info=True)
        if self._pcie_ok:
            try:
                tx = n.nvmlDeviceGetPcieThroughput(h, n.NVML_PCIE_UTIL_TX_BYTES)  # KB/s
                rx = n.nvmlDeviceGetPcieThroughput(h, n.NVML_PCIE_UTIL_RX_BYTES)
                s.pcie_tx_gbs.append(tx * 1024 / 1e9)
                s.pcie_rx_gbs.append(rx * 1024 / 1e9)
            except Exception:  # noqa: BLE001
                self._pcie_ok = False
        if self._proc_ok:
            try:
                procs = n.nvmlDeviceGetProcessUtilization(h, self._last_proc_ts)
                if procs:
                    self._last_proc_ts = max(p.timeStamp for p in procs)
                    s.sm_util.append(float(sum(p.smUtil for p in procs)))
            except Exception:  # noqa: BLE001
                self._proc_ok = False

    def _loop(self) -> None:
        """Sampler thread body."""
        while not self._stop.is_set():
            try:
                self._sample()
            except Exception:
                logger.debug("GPU sampling error", exc_info=True)
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        """Start sampling (no-op without NVML)."""
        if not self.available or self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="gpu-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop sampling and join the thread."""
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=5)
        self._thread = None

    # ------------------------------------------------------------------ report
    def energy_wh(self) -> float | None:
        """Integrate power over time (trapezoidal rule).

        Returns:
            Energy in Wh, or ``None`` with fewer than two samples.
        """
        s = self.samples
        n = min(len(s.t), len(s.power_w))
        if n < 2:
            return None
        joules = sum((s.t[i] - s.t[i - 1]) * (s.power_w[i] + s.power_w[i - 1]) / 2 for i in range(1, n))
        return joules / 3600.0

    def metrics_rows(self, stats: RunStats, data_rate_factor: float = 2.0) -> tuple[list[list[Any]], str]:
        """Build the 20-row ``GPU_Metrics`` table.

        Args:
            stats: Workload statistics for this model run.
            data_rate_factor: Multiplier in the bandwidth formula (2 for DDR as reported by NVML).

        Returns:
            ``(rows, top_note)``; rows are ``[#, Parameter, Value, Unit, Method, Notes]``.
        """
        note = OPENAI_NOTE if stats.provider == "openai" else ""
        if not self.available:
            return [[1, "GPU", "No GPU detected", "", "N/A", self.error or "NVML not available"]], note
        return build_metrics_rows(
            self.static, self.samples, lookup_gpu_spec(self.static.name), stats, self.energy_wh(), data_rate_factor
        ), note


def dcgm_sm_active() -> float | None:
    """Read SM activity once via DCGM (``dcgmi dmon -e 1002``) if DCGM is installed.

    Returns:
        SM active percentage, or ``None`` when DCGM is absent or fails.
    """
    import shutil
    import subprocess

    if not shutil.which("dcgmi"):
        return None
    try:
        out = subprocess.run(
            ["dcgmi", "dmon", "-e", "1002", "-c", "1"], capture_output=True, text=True, timeout=10, check=False
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in reversed(out.splitlines()):
        parts = line.split()
        if parts and parts[0].startswith("GPU"):
            try:
                return float(parts[-1]) * 100
            except ValueError:
                return None
    return None


def build_metrics_rows(
    st: GPUStatic,
    sm: Samples,
    spec: GPUSpec | None,
    stats: RunStats,
    energy_wh: float | None,
    data_rate_factor: float = 2.0,
) -> list[list[Any]]:
    """Assemble the 20 GPU parameters (pure function; testable without a GPU).

    Args:
        st: Static device facts.
        sm: Collected samples.
        spec: Spec-table entry or ``None``.
        stats: Workload statistics.
        energy_wh: Integrated energy (Wh).
        data_rate_factor: DDR factor for the bandwidth formula.

    Returns:
        Rows ``[#, Parameter, Value, Unit, Method, Notes]``.
    """
    no_spec = "N/A — GPU not in spec table"
    spec_src = f"Spec lookup ({spec.name})" if spec else "N/A"
    rows: list[list[Any]] = []

    def add(param: str, value: str, unit: str, method: str, notes: str = "") -> None:
        """Append one row.

        Args:
            param: Parameter name.
            value: Value text.
            unit: Unit.
            method: How it was obtained.
            notes: Formula / caveats.
        """
        rows.append([len(rows) + 1, param, value, unit, method, notes])

    nsamp = len(sm.t)
    sampled = f"{nsamp} samples @1 s during translation"
    add(
        "GPU Utilization",
        f"avg {_fmt(_avg(sm.util), 1)} / peak {_fmt(max(sm.util) if sm.util else None, 1)}",
        "%",
        "Measured: NVML utilization.gpu",
        sampled,
    )
    add(
        "GPU Memory (VRAM) Used",
        f"avg {_fmt(_avg(sm.mem_used_gb))} / peak {_fmt(max(sm.mem_used_gb) if sm.mem_used_gb else None)}",
        "GB",
        "Measured: NVML memory info",
        "Device-wide (all processes, incl. Ollama server)",
    )
    add("Total VRAM Capacity", _fmt(st.total_mem_gb), "GB", "Measured: NVML", st.name)
    add(
        "GPU Core Clock Speed",
        f"avg {_fmt(_avg(sm.gfx_clock), 0)} / max {_fmt(st.max_graphics_clock, 0)}",
        "MHz",
        "Measured: NVML graphics clock (avg) + max clock",
    )
    add(
        "Memory Clock Speed",
        f"avg {_fmt(_avg(sm.mem_clock), 0)} / max {_fmt(st.max_mem_clock, 0)}",
        "MHz",
        "Measured: NVML memory clock",
    )
    bw_theory = None
    if st.max_mem_clock and st.bus_width:
        bw_theory = st.max_mem_clock * 1e6 * data_rate_factor * st.bus_width / 8 / 1e9
        cross = f"; spec sheet {spec.mem_bandwidth_gbs:,.0f} GB/s" if spec else ""
        add(
            "Memory Bandwidth",
            _fmt(bw_theory, 1),
            "GB/s",
            "Derived",
            f"max mem clock {st.max_mem_clock:.0f} MHz × {data_rate_factor:g} × {st.bus_width}-bit ÷ 8{cross}",
        )
    elif spec:
        bw_theory = spec.mem_bandwidth_gbs
        add("Memory Bandwidth", _fmt(bw_theory, 0), "GB/s", spec_src, "NVML bus width not exposed")
    else:
        add("Memory Bandwidth", "N/A", "GB/s", "N/A", "Bus width not exposed and " + no_spec)
    if st.num_cores:
        add("CUDA Cores / Stream Processors", str(st.num_cores), "Count", "Measured: NVML nvmlDeviceGetNumGpuCores")
    else:
        add(
            "CUDA Cores / Stream Processors",
            str(spec.cuda_cores) if spec else no_spec,
            "Count",
            spec_src,
            "nvmlDeviceGetNumGpuCores not supported by this driver",
        )
    add("Tensor Cores / AI Accelerators", str(spec.tensor_cores) if spec else no_spec, "Count", spec_src)
    add("FP32 Compute Throughput", _fmt(spec.fp32_tflops, 1) if spec else no_spec, "TFLOPS", spec_src, "Peak")
    add(
        "FP16/BF16 Compute Throughput",
        _fmt(spec.fp16_tflops, 1) if spec else no_spec,
        "TFLOPS",
        spec_src,
        (spec.note if spec else "") + "; dense tensor peak",
    )
    add(
        "INT8 Compute Throughput",
        (_fmt(spec.int8_tops, 1) if spec.int8_tops is not None else "N/A — not published") if spec else no_spec,
        "TOPS",
        spec_src,
        "Dense tensor peak",
    )
    add("Batch Size", stats.batch_size_desc or "N/A", "Samples per batch", "From config/run")
    dur = stats.duration_s
    rps = stats.records_processed / dur if dur > 0 else None
    tps = stats.output_tokens / dur if dur > 0 else None
    add(
        "Throughput",
        f"{_fmt(rps, 3)} rec/s; {_fmt(tps, 1)} output tok/s",
        "records/s and output tokens/s",
        "Measured over the run",
        f"{stats.records_processed} live records, {dur:.1f} s translation phase",
    )
    lat = stats.latencies
    add(
        "Latency",
        f"mean {_fmt(_avg(lat))} / p50 {_fmt(percentile(lat, 50))} / p95 {_fmt(percentile(lat, 95))}",
        "s",
        "Measured per record",
        "Batched seq2seq: batch time ÷ batch size" if stats.provider in ("hf_seq2seq", "custom") else "",
    )
    pcie_theory = None
    if st.pcie_gen_max and st.pcie_width_max:
        pcie_theory = PCIE_GBPS_PER_LANE.get(st.pcie_gen_max, 0) * st.pcie_width_max
    elif spec:
        pcie_theory = PCIE_GBPS_PER_LANE.get(spec.pcie_gen, 0) * spec.pcie_width
    if sm.pcie_tx_gbs:
        val = (
            f"TX avg {_fmt(_avg(sm.pcie_tx_gbs), 3)} / peak {_fmt(max(sm.pcie_tx_gbs), 3)}; "
            f"RX avg {_fmt(_avg(sm.pcie_rx_gbs), 3)} / peak {_fmt(max(sm.pcie_rx_gbs), 3)}; "
            f"theoretical max {_fmt(pcie_theory, 1)}"
        )
        add(
            "PCIe / NVLink Bandwidth",
            val,
            "GB/s",
            "Measured: NVML PCIe TX/RX throughput",
            f"link Gen{st.pcie_gen_cur}×{st.pcie_width_cur} (max Gen{st.pcie_gen_max}×{st.pcie_width_max}); "
            "NVLink not present on Colab single-GPU VMs",
        )
    else:
        add(
            "PCIe / NVLink Bandwidth",
            f"theoretical max {_fmt(pcie_theory, 1)}",
            "GB/s",
            "Derived: link gen × width",
            "NVML PCIe throughput counters not exposed",
        )
    add(
        "Power Consumption",
        f"avg {_fmt(_avg(sm.power_w), 1)} / peak {_fmt(max(sm.power_w) if sm.power_w else None, 1)}; "
        f"energy {_fmt(energy_wh, 4)} Wh",
        "W",
        "Measured: NVML power usage",
        "Energy = ∫P dt (trapezoidal)",
    )
    add(
        "GPU Temperature",
        f"avg {_fmt(_avg(sm.temp_c), 1)} / peak {_fmt(max(sm.temp_c) if sm.temp_c else None, 1)}",
        "°C",
        "Measured: NVML",
    )
    avg_p = _avg(sm.power_w)
    tps_per_w = (tps / avg_p) if (tps is not None and avg_p) else None
    tpj = (stats.total_tokens / (energy_wh * 3600)) if energy_wh else None
    add(
        "Energy Efficiency",
        f"{_fmt(tps_per_w, 4)} tok/s per W; {_fmt(tpj, 4)} tokens per joule",
        "tokens/s per W, tokens per joule",
        "Derived",
        "output tok/s ÷ avg W; (input+output tokens) ÷ energy J"
        + ("; inference ran remotely, so this is not meaningful" if stats.provider == "openai" else ""),
    )
    mem_util_avg = _avg(sm.mem_util)
    rw = (mem_util_avg / 100 * bw_theory) if (mem_util_avg is not None and bw_theory) else None
    add(
        "GPU Memory Read/Write Throughput",
        _fmt(rw, 1),
        "GB/s",
        "Derived (approx.)",
        "NVML memory-controller utilisation % × theoretical bandwidth (NVML exposes busy-time %, not bytes)",
    )
    if sm.sm_util:
        add(
            "Compute Occupancy / SM Utilization",
            f"avg {_fmt(_avg(sm.sm_util), 1)} / peak {_fmt(max(sm.sm_util), 1)}",
            "%",
            "Measured: NVML process SM utilisation",
        )
    elif (dcgm := dcgm_sm_active()) is not None:
        add(
            "Compute Occupancy / SM Utilization",
            _fmt(dcgm, 1),
            "%",
            "Measured: DCGM SM_ACTIVE (field 1002)",
            "Single reading at end of run",
        )
    else:
        add(
            "Compute Occupancy / SM Utilization",
            f"N/A — not exposed on Colab (proxy: GPU utilisation avg {_fmt(_avg(sm.util), 1)})",
            "%",
            "N/A",
            "nvmlDeviceGetProcessUtilization unsupported; DCGM not present",
        )
    return rows
