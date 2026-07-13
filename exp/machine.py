from __future__ import annotations

import os
import platform
import shutil
import socket
import subprocess
import time
from typing import Any

# Machine-state collection for the web UI. Everything here is best-effort:
# psutil and NVIDIA tooling are optional, and each probe degrades to None /
# an empty list instead of raising, so the endpoint works on any machine
# (CUDA training box, Apple Silicon dev laptop, CI, ...).

_cache: dict[str, Any] = {'time': 0.0, 'state': None}

def _get_cpu_and_memory() -> dict[str, Any]:
	''' Collects CPU and memory statistics, preferring psutil with a stdlib fallback.

	Returns: stats: dict  keys: cpu_percent, cpu_count, load_avg, mem_total, mem_used, mem_percent (missing probes are None)
	'''
	stats: dict[str, Any] = {
		'cpu_percent': None, 'cpu_count': os.cpu_count(), 'load_avg': None,
		'mem_total': None, 'mem_used': None, 'mem_percent': None,
	}
	try:
		stats['load_avg'] = list(os.getloadavg())
	except OSError:
		pass
	try:
		import psutil
	except ImportError:
		return stats
	# interval=None returns utilisation since the previous call — ideal for polling
	stats['cpu_percent'] = psutil.cpu_percent(interval=None)
	mem = psutil.virtual_memory()
	stats['mem_total'] = mem.total
	stats['mem_used'] = mem.total - mem.available
	stats['mem_percent'] = mem.percent
	return stats

def _get_gpus_pynvml() -> list[dict[str, Any]] | None:
	''' Collects per-GPU statistics via pynvml.

	Returns: gpus: list[dict] | None  one dict per GPU, or None if pynvml is unavailable/fails
	'''
	try:
		import pynvml
	except ImportError:
		return None
	try:
		pynvml.nvmlInit()
	except Exception:
		return None
	try:
		gpus = []
		for i in range(pynvml.nvmlDeviceGetCount()):
			handle = pynvml.nvmlDeviceGetHandleByIndex(i)
			name = pynvml.nvmlDeviceGetName(handle)
			name = name.decode('utf-8') if isinstance(name, bytes) else name
			util = pynvml.nvmlDeviceGetUtilizationRates(handle)
			mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
			gpu: dict[str, Any] = {
				'index': i, 'name': name,
				'util_percent': util.gpu,
				'mem_total': mem.total, 'mem_used': mem.used,
				'temperature': None, 'power_draw': None, 'power_limit': None,
			}
			try:
				gpu['temperature'] = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
			except Exception:
				pass
			try:
				gpu['power_draw'] = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
				gpu['power_limit'] = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
			except Exception:
				pass
			gpus.append(gpu)
		return gpus
	except Exception:
		return None
	finally:
		try:
			pynvml.nvmlShutdown()
		except Exception:
			pass

def _get_gpus_nvidia_smi() -> list[dict[str, Any]] | None:
	''' Collects per-GPU statistics by shelling out to nvidia-smi.

	Returns: gpus: list[dict] | None  one dict per GPU, or None if nvidia-smi is unavailable/fails
	'''
	if shutil.which('nvidia-smi') is None:
		return None
	query = 'index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit'
	try:
		out = subprocess.run(
			['nvidia-smi', f'--query-gpu={query}', '--format=csv,noheader,nounits'],
			capture_output=True, text=True, timeout=3.0)
	except (subprocess.SubprocessError, OSError):
		return None
	if out.returncode != 0:
		return None

	def _num(raw: str) -> float | None:
		try:
			return float(raw)
		except ValueError:
			return None # nvidia-smi prints '[N/A]' for unsupported fields

	gpus = []
	for line in out.stdout.strip().splitlines():
		parts = [part.strip() for part in line.split(',')]
		if len(parts) < 8:
			continue
		mem_used, mem_total = _num(parts[3]), _num(parts[4])
		gpus.append({
			'index': int(parts[0]), 'name': parts[1],
			'util_percent': _num(parts[2]),
			# nvidia-smi reports memory in MiB
			'mem_used': int(mem_used * 1024 * 1024) if mem_used is not None else None,
			'mem_total': int(mem_total * 1024 * 1024) if mem_total is not None else None,
			'temperature': _num(parts[5]),
			'power_draw': _num(parts[6]), 'power_limit': _num(parts[7]),
		})
	return gpus

def get_machine_state(master_directory: str | None = None, cache_seconds: float = 2.0) -> dict[str, Any]:
	''' Returns a snapshot of the machine state for the web UI, cached to keep polling cheap.

	Args:
		1. master_directory: str | None  [None] the experiments directory whose disk partition to report; defaults to the working directory
		2. cache_seconds: float  [2.0] reuse a snapshot younger than this many seconds
	Returns: state: dict  keys: hostname, platform, time, cpu{...}, gpus[...], disk{total, used, free}
	'''
	now = time.time()
	if _cache['state'] is not None and now - _cache['time'] < cache_seconds:
		return _cache['state']
	disk_path = master_directory if master_directory and os.path.exists(master_directory) else os.getcwd()
	try:
		usage = shutil.disk_usage(disk_path)
		disk = {'total': usage.total, 'used': usage.used, 'free': usage.free}
	except OSError:
		disk = {'total': None, 'used': None, 'free': None}
	gpus = _get_gpus_pynvml()
	if gpus is None:
		gpus = _get_gpus_nvidia_smi()
	state = {
		'hostname': socket.gethostname(),
		'platform': platform.platform(terse=True),
		'time': now,
		'cpu': _get_cpu_and_memory(),
		'gpus': gpus if gpus is not None else [],
		'disk': disk,
	}
	_cache['time'] = now
	_cache['state'] = state
	return state
