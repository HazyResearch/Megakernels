"""Convert LlamaKernels timing tensors to .mkprof binary format.
(see megakernels/timings.py in megakernels high throughput)

Input:  3D tensor (num_sms, max_instructions, 16) — cycle counts from clock64()
Output: .mkprof binary loadable by util/profiler/index.html

The 16-slot timing data is remapped into a 128-slot format expected by the viewer:
  0-1: metadata (icode, sm_id)
  5-6: controller events (packed)
  8-15: functional unit special_events (raw ns, drives bar rendering)
  16-47/48-79/112-127: loader/consumer/storer events (packed, contiguous)
"""

import json
from pathlib import Path
import numpy as np, torch

# Timing event indices (csrc/utils.cuh)
IC, GW, GWD, FL, LL, FU, LU, FS, LS, CS, OR, CW, CWD, GS, GSD = range(15)

# Packed event type codes (lower 4 bits)
E_LOAD, E_COMPUTE, E_STORE, E_WAIT, E_READY = 0, 2, 3, 5, 6

TIMING_LEN, INST_LEN = 128, 32


def _pack(ns: int, etype: int) -> int:
    """Pack ns timestamp + 4-bit event type. Viewer: ts=(raw>>>4)*16, type=raw&0xF."""
    if ns <= 0: return 0
    return ((max(1, ns // 16) & 0x0FFFFFFF) << 4) | (etype & 0xF)


def _first_pos(*vals: int) -> int:
    for v in vals:
        if v > 0: return v
    return 0


def _pack_events(pairs, base, ev, ev_base):
    """Sort (ns, etype) pairs by time and write contiguously at ev[ev_base+base:]."""
    packed = sorted(((ns, et) for ns, et in pairs if ns > 0), key=lambda x: x[0])
    for j, (ns, et) in enumerate(packed):
        ev[ev_base + base + j] = np.uint32(_pack(ns, et))


def get_llama1b_config():
    return {
        "format_version": "1.2",
        "instruction_types": {
            "0": {"name": "No Op",                "color": "#808080", "params": {}},
            "1": {"name": "RMS QKV Rope Append",  "color": "#ff7f0e", "params": {"1": "layer_idx", "2": "start_block", "3": "end_block"}},
            "2": {"name": "Partial Attention",     "color": "#2ca02c", "params": {"1": "layer_idx", "2": "kv_head"}},
            "3": {"name": "O Proj Residual",       "color": "#d62728", "params": {"1": "layer_idx", "2": "start_block", "3": "end_block"}},
            "4": {"name": "RMS Upgate SiLU",       "color": "#9467bd", "params": {"1": "layer_idx", "2": "start_block", "3": "end_block"}},
            "5": {"name": "Down Proj Residual",    "color": "#7f7f7f", "params": {"1": "layer_idx", "2": "start_block", "3": "end_block"}},
            "6": {"name": "RMS LM Head",           "color": "#17becf", "params": {"1": "start_block", "2": "end_block"}},
        },
        "instruction_format": {"instruction_length": INST_LEN, "timing_length": TIMING_LEN},
        "functional_units": {
            "0": {"name": "Loader",   "event_range": {"start": 16,  "end": 47},  "height_multiplier": 1.0, "special_events": {"start": 8,  "end": 9}},
            "1": {"name": "Consumer", "event_range": {"start": 48,  "end": 79},  "height_multiplier": 2.0, "special_events": {"start": 12, "end": 13}},
            "2": {"name": "Storer",   "event_range": {"start": 112, "end": 127}, "height_multiplier": 1.0, "special_events": {"start": 14, "end": 15}},
            "3": {"name": "Launcher", "event_range": {"start": 80,  "end": 111}, "height_multiplier": 1.0, "special_events": {"start": 10, "end": 11}},
        },
        "main_functional_unit": 1,
        "event_types": {
            "0": {"name": "LOAD_EVENT",    "color": "#0000ff"},
            "2": {"name": "COMPUTE_EVENT", "color": "#aa00ff"},
            "3": {"name": "STORE_EVENT",   "color": "#ffff00"},
            "5": {"name": "WAIT_EVENT",    "color": "#ff0000"},
            "6": {"name": "READY_EVENT",   "color": "#00ff00"},
        },
        "controller_events": {
            "5": {"name": "CTRL_WAIT_START", "color": "#FA8072"},
            "6": {"name": "CTRL_WAIT_END",   "color": "#32CD32"},
        },
        "num_gpus": 1, "total_processors": 0, "max_instructions": 0,
        "time_unit_flag": 1, "has_events_flag": 1,
    }


def timings_to_mkprof(timings: torch.Tensor, output: str, clock_mhz: float = 2100.0,
                       config: dict | None = None):
    output = str(Path(output).with_suffix(".mkprof"))
    config = config or get_llama1b_config()
    if timings.is_cuda: timings = timings.cpu()

    t = timings.numpy()
    num_sms, max_instrs, tw = t.shape

    # Normalize: subtract min cycle so ns values stay small
    workers = t[:, :, 1:15]
    nz = workers[workers > 0]
    min_c = int(nz.min()) if nz.size else 0

    total = num_sms * max_instrs
    instructions = np.zeros(total * INST_LEN, dtype=np.int32)
    starts = np.zeros(total, dtype=np.uint32)
    ends = np.zeros(total, dtype=np.uint32)
    ev = np.zeros(total * TIMING_LEN, dtype=np.uint32)

    for sm in range(num_sms):
        for ii in range(max_instrs):
            raw = t[sm, ii]
            icode = int(raw[IC])
            if icode <= 0: continue

            ns = lambda s: max(0, int((int(raw[s]) - min_c) * 1000.0 / clock_mhz)) if int(raw[s]) > 0 else 0
            gw, gwd = ns(GW), ns(GWD)
            fl, ll  = ns(FL), ns(LL)
            fu, lu  = ns(FU), ns(LU)
            fs, ls  = ns(FS), ns(LS)
            cs, orr = ns(CS), ns(OR)
            cw, cwd = ns(CW), ns(CWD)
            gs, gsd = ns(GS), ns(GSD)

            all_ns = [v for v in (gw, gwd, fl, ll, fu, lu, fs, ls, cs, orr, gs, gsd) if v > 0]
            if len(all_ns) < 2: continue
            t_start, t_end = min(all_ns), max(all_ns)
            if t_end <= t_start: continue

            idx = sm * max_instrs + ii
            instructions[idx * INST_LEN] = icode
            starts[idx] = np.uint32(t_start)
            ends[idx] = np.uint32(t_end)

            b = idx * TIMING_LEN
            ev[b], ev[b+1] = np.uint32(icode), np.uint32(sm)
            ev[b+5] = np.uint32(_pack(cw, E_WAIT))
            ev[b+6] = np.uint32(_pack(cwd, E_READY))
            ev[b+8]  = np.uint32(_first_pos(gw, fl))
            ev[b+9]  = np.uint32(_first_pos(ll, gwd))
            ev[b+12] = np.uint32(_first_pos(cs, fu))
            ev[b+13] = np.uint32(_first_pos(orr, lu))
            ev[b+14] = np.uint32(fs)
            ev[b+15] = np.uint32(ls)

            _pack_events([(gw, E_WAIT), (gwd, E_READY), (fl, E_LOAD), (ll, E_LOAD)], 16, ev, b)
            _pack_events([(cs, E_COMPUTE), (fu, E_COMPUTE), (lu, E_COMPUTE), (orr, E_READY)], 48, ev, b)
            _pack_events([(fs, E_STORE), (ls, E_STORE), (gs, E_STORE), (gsd, E_STORE)], 112, ev, b)

    config["total_processors"] = num_sms
    config["max_instructions"] = max_instrs

    with open(output, "wb") as f:
        f.write(b"MKPROF1.2\n")
        f.write(json.dumps(config, separators=(",", ":")).encode())
        pad = (4 - (f.tell() % 4)) % 4
        if pad: f.write(b"\x00" * pad)
        for arr in (instructions, starts, ends, ev):
            f.write(arr.tobytes())

    sz = Path(output).stat().st_size
    print(f"Written {output}: {sz:,} bytes ({sz/1024/1024:.1f} MB)")
