from __future__ import annotations

import socket
import struct
import time
from functools import lru_cache
from pathlib import Path
from typing import Iterator

# A dependency-free reader/writer for TensorBoard scalar event files.
#
# TensorBoard event files are TFRecord streams of serialized `Event` protobufs:
#   uint64  length (little-endian)
#   uint32  masked crc32c of the length bytes
#   bytes   data[length]           (the Event protobuf)
#   uint32  masked crc32c of data
#
# We hand-roll the tiny subset of protobuf wire format needed for scalars so the
# web server can plot live metrics without requiring the `tensorboard` package.
# Relevant proto fields:
#   Event:         1 wall_time (double)  2 step (int64)  3 file_version (string)  5 summary (Summary)
#   Summary:       1 value (repeated Summary.Value)
#   Summary.Value: 1 tag (string)  2 simple_value (float)  8 tensor (TensorProto)
#   TensorProto:   4 tensor_content (bytes)  5 float_val (repeated float)

_CRC_MASK_DELTA = 0xa282ead8

@lru_cache(maxsize=1)
def _crc32c_table() -> tuple[int, ...]:
	''' Returns the 256-entry lookup table for CRC-32C (Castagnoli, reflected polynomial 0x82F63B78).

	Returns: table: tuple[int, ...]  [uint32, (256,)] the CRC lookup table
	'''
	table = []
	for i in range(256):
		crc = i
		for _ in range(8):
			crc = (crc >> 1) ^ 0x82F63B78 if crc & 1 else crc >> 1
		table.append(crc)
	return tuple(table)

def _masked_crc32c(data: bytes) -> int:
	''' Computes the masked CRC-32C used by the TFRecord framing.

	Args:
		1. data: bytes  the payload to checksum
	Returns: crc: int  [uint32] the masked checksum
	'''
	table = _crc32c_table()
	crc = 0xFFFFFFFF
	for byte in data:
		crc = table[(crc ^ byte) & 0xFF] ^ (crc >> 8)
	crc ^= 0xFFFFFFFF
	# rotate right by 15 bits and add the mask delta
	masked = (((crc >> 15) | (crc << 17)) + _CRC_MASK_DELTA) & 0xFFFFFFFF
	return masked

### protobuf wire-format primitives ###

def _encode_varint(value: int) -> bytes:
	''' Encodes a non-negative integer as a protobuf varint.

	Args:
		1. value: int  [>= 0] the integer to encode
	Returns: encoded: bytes  the varint encoding
	'''
	if value < 0:
		raise ValueError(f'Cannot varint-encode negative value {value}.')
	out = bytearray()
	while True:
		bits = value & 0x7F
		value >>= 7
		if value:
			out.append(bits | 0x80)
		else:
			out.append(bits)
			return bytes(out)

def _decode_varint(buf: bytes, pos: int) -> tuple[int, int]:
	''' Decodes a protobuf varint from buf starting at pos.

	Args:
		1. buf: bytes  the buffer to read from
		2. pos: int  the offset to start reading at
	Returns: (value, new_pos): tuple[int, int]  the decoded unsigned value and the offset just past it
	'''
	result = 0
	shift = 0
	while True:
		if pos >= len(buf):
			raise ValueError('Truncated varint.')
		byte = buf[pos]
		pos += 1
		result |= (byte & 0x7F) << shift
		if not byte & 0x80:
			return result, pos
		shift += 7
		if shift >= 70: # int64 varints are at most 10 bytes
			raise ValueError('Varint too long.')

def _iter_fields(buf: bytes) -> Iterator[tuple[int, int, int | bytes]]:
	''' Iterates over the top-level fields of a serialized protobuf message.

	Args:
		1. buf: bytes  the serialized message
	Yields: (field_number, wire_type, value): tuple[int, int, int | bytes]  value is an int for varint/fixed types (raw bits) and bytes for length-delimited fields
	'''
	pos = 0
	while pos < len(buf):
		key, pos = _decode_varint(buf, pos)
		field, wire = key >> 3, key & 0x7
		if wire == 0: # varint
			value, pos = _decode_varint(buf, pos)
		elif wire == 1: # fixed64
			value = int.from_bytes(buf[pos:pos + 8], 'little')
			pos += 8
		elif wire == 2: # length-delimited
			length, pos = _decode_varint(buf, pos)
			value = buf[pos:pos + length]
			pos += length
		elif wire == 5: # fixed32
			value = int.from_bytes(buf[pos:pos + 4], 'little')
			pos += 4
		else:
			raise ValueError(f'Unsupported protobuf wire type {wire}.')
		if pos > len(buf):
			raise ValueError('Truncated field.')
		yield field, wire, value

def _bits_to_double(bits: int) -> float:
	return struct.unpack('<d', bits.to_bytes(8, 'little'))[0]

def _bits_to_float(bits: int) -> float:
	return struct.unpack('<f', bits.to_bytes(4, 'little'))[0]

### reading ###

def _tensor_first_float(buf: bytes) -> float | None:
	''' Best-effort extraction of the first float from a serialized TensorProto (new-style scalar summaries).

	Args:
		1. buf: bytes  the serialized TensorProto
	Returns: value: float | None  the first float value, or None if not found
	'''
	for field, wire, value in _iter_fields(buf):
		if field == 5: # float_val: repeated float — packed (bytes) or unpacked (fixed32)
			if wire == 2 and len(value) >= 4:
				return struct.unpack('<f', value[:4])[0]
			if wire == 5:
				return _bits_to_float(value)
		if field == 4 and wire == 2 and len(value) >= 4: # tensor_content: raw little-endian bytes
			return struct.unpack('<f', value[:4])[0]
	return None

def iter_scalar_events(path: str | Path) -> Iterator[tuple[float, int, str, float]]:
	''' Iterates over all scalar summaries in a TensorBoard event file, stopping cleanly at a truncated tail (safe on live files).

	Args:
		1. path: str | Path  the event file to read
	Yields: (wall_time, step, tag, value): tuple[float, int, str, float]  one scalar datapoint per summary value
	'''
	with open(path, 'rb') as f:
		data = f.read()
	pos = 0
	while pos + 12 <= len(data):
		length = int.from_bytes(data[pos:pos + 8], 'little')
		# full record = 8 (len) + 4 (len crc) + length (payload) + 4 (payload crc)
		end = pos + 12 + length + 4
		if end > len(data):
			break # truncated tail — writer is mid-record
		event = data[pos + 12:pos + 12 + length]
		pos = end
		try:
			wall_time, step, summary = 0.0, 0, None
			for field, wire, value in _iter_fields(event):
				if field == 1 and wire == 1:
					wall_time = _bits_to_double(value)
				elif field == 2 and wire == 0:
					# int64 varint: convert from two's complement
					step = value - (1 << 64) if value >= (1 << 63) else value
				elif field == 5 and wire == 2:
					summary = value
			if summary is None:
				continue
			for field, wire, value in _iter_fields(summary):
				if field != 1 or wire != 2:
					continue
				tag, scalar = None, None
				for vfield, vwire, vvalue in _iter_fields(value):
					if vfield == 1 and vwire == 2:
						tag = vvalue.decode('utf-8', errors='replace')
					elif vfield == 2 and vwire == 5:
						scalar = _bits_to_float(vvalue)
					elif vfield == 8 and vwire == 2 and scalar is None:
						scalar = _tensor_first_float(vvalue)
				if tag is not None and scalar is not None:
					yield wall_time, step, tag, scalar
		except ValueError:
			continue # skip malformed records rather than dying mid-file

def find_event_files(directory: str | Path) -> list[Path]:
	''' Recursively finds TensorBoard event files under an experiment directory.

	Args:
		1. directory: str | Path  the experiment directory (event files may live in version_N subfolders)
	Returns: files: list[Path]  event file paths sorted by modification time (oldest first)
	'''
	directory = Path(directory)
	if not directory.is_dir():
		return []
	files = [p for p in directory.rglob('events.out.tfevents.*') if p.is_file()]
	return sorted(files, key=lambda p: p.stat().st_mtime)

def read_scalars(directory: str | Path, tags: set[str] | None = None) -> dict[str, list[tuple[int, float, float]]]:
	''' Reads all scalar series from every event file under an experiment directory, merged and sorted by wall time.

	Args:
		1. directory: str | Path  the experiment directory
		2. tags: set[str] | None  [None] if given, only collect these tags
	Returns: scalars: dict[str, list[tuple[int, float, float]]]  tag -> list of (step, wall_time, value) datapoints
	'''
	scalars: dict[str, list[tuple[int, float, float]]] = {}
	for path in find_event_files(directory):
		for wall_time, step, tag, value in iter_scalar_events(path):
			if tags is not None and tag not in tags:
				continue
			scalars.setdefault(tag, []).append((step, wall_time, value))
	for series in scalars.values():
		series.sort(key=lambda point: (point[1], point[0]))
	return scalars

### writing ###

class ScalarEventWriter:
	''' A minimal TensorBoard-compatible event file writer for scalar metrics (used by exp/demo.py to fake training runs).

	Properties:
		1. path: Path  the event file being written
	'''

	def __init__(self, logdir: str | Path, wall_time: float | None = None):
		''' Creates the log directory and opens a new event file with the standard file-version header record.

		Args:
			1. logdir: str | Path  directory to create the event file in
			2. wall_time: float | None  [None] creation timestamp, defaults to now
		'''
		logdir = Path(logdir)
		logdir.mkdir(parents=True, exist_ok=True)
		wall_time = time.time() if wall_time is None else wall_time
		hostname = socket.gethostname()
		self.path = logdir / f'events.out.tfevents.{int(wall_time)}.{hostname}'
		self._file = open(self.path, 'wb')
		# header record: Event{wall_time, file_version: "brain.Event:2"}
		version = b'brain.Event:2'
		event = (_encode_varint(1 << 3 | 1) + struct.pack('<d', wall_time)
				+ _encode_varint(3 << 3 | 2) + _encode_varint(len(version)) + version)
		self._write_record(event)
		self.flush()

	def _write_record(self, payload: bytes):
		length_bytes = struct.pack('<Q', len(payload))
		self._file.write(length_bytes)
		self._file.write(struct.pack('<I', _masked_crc32c(length_bytes)))
		self._file.write(payload)
		self._file.write(struct.pack('<I', _masked_crc32c(payload)))

	def add_scalar(self, tag: str, value: float, step: int, wall_time: float | None = None):
		''' Appends one scalar datapoint and flushes so readers can tail the file live.

		Args:
			1. tag: str  the metric name (eg, val_loss)
			2. value: float  the scalar value
			3. step: int  [>= 0] the global step
			4. wall_time: float | None  [None] timestamp, defaults to now
		'''
		wall_time = time.time() if wall_time is None else wall_time
		tag_bytes = tag.encode('utf-8')
		value_msg = (_encode_varint(1 << 3 | 2) + _encode_varint(len(tag_bytes)) + tag_bytes
					+ _encode_varint(2 << 3 | 5) + struct.pack('<f', value))
		summary = _encode_varint(1 << 3 | 2) + _encode_varint(len(value_msg)) + value_msg
		event = (_encode_varint(1 << 3 | 1) + struct.pack('<d', wall_time)
				+ _encode_varint(2 << 3 | 0) + _encode_varint(step)
				+ _encode_varint(5 << 3 | 2) + _encode_varint(len(summary)) + summary)
		self._write_record(event)
		self.flush()

	def flush(self):
		self._file.flush()

	def close(self):
		if not self._file.closed:
			self._file.close()

	def __enter__(self) -> 'ScalarEventWriter':
		return self

	def __exit__(self, *exc_info):
		self.close()
