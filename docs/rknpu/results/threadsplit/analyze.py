#!/usr/bin/env python3
"""Turn run-*.log into the tables used by report.md.

Reporting rule, pre-registered before the timed runs:
  * statistic = median over all timed repetitions of a variant, measured with
    all variants interleaved within one round;
  * a rep is called a *tick spike* if it exceeds the variant's median by more
    than 3.33 ms, which is one tick at CONFIG_HZ=300 on this host;
  * spikes are counted and their maximum reported, but no rep is discarded.
"""
import sys, statistics, re, collections

SIZES = [131072, 1179648]

def load(path):
    rows = collections.defaultdict(list)   # (phase, variant, size) -> [ns]
    order = []
    phase_of = {}
    cur_phase = "?"
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            m = re.match(r"^# phase (\S+) n=(\d+)", line)
            if m:
                cur_phase = m.group(1)
                continue
            if line.startswith("#") or not line:
                continue
            p = line.split(",")
            if len(p) != 6 or p[0] in ("variant",):
                continue
            var, size, rep, ns = p[0], int(p[1]), int(p[2]), float(p[3])
            key = (cur_phase, var, size)
            if key not in rows:
                order.append(key)
            rows[key].append(ns)
            phase_of[var] = cur_phase
    return rows, order

def stats(v):
    med = statistics.median(v)
    spikes = [x for x in v if x > med + 3.33e6]
    return dict(n=len(v), med=med, mn=min(v), mx=max(v), nspike=len(spikes),
                spike_max=max(spikes) if spikes else 0.0)

def fmt(x):
    return f"{x:.4f}"

def main(paths):
    data = {}
    for p in paths:
        rows, order = load(p)
        data[p] = rows
    # primary run = first path
    primary = data[paths[0]]

    print(f"## per-variant statistics ({paths[0]})\n")
    print("| phase | variant | N | reps | median ms | min ms | max ms | tick spikes |")
    print("| --- | --- | --: | --: | --: | --: | --: | --: |")
    for (ph, var, size) in [k for k in primary]:
        s = stats(primary[(ph, var, size)])
        print(f"| {ph} | {var} | {size} | {s['n']} | {s['med']/1e6:.4f} | "
              f"{s['mn']/1e6:.4f} | {s['mx']/1e6:.4f} | {s['nspike']} |")

    print(f"\n## run-to-run agreement (median ms)\n")
    print("| phase | variant | N | run a | run b | diff % |")
    print("| --- | --- | --: | --: | --: | --: |")
    a, b = data[paths[0]], data[paths[1]]
    worst = 0.0
    for k in a:
        if k not in b:
            continue
        ma = statistics.median(a[k]); mb = statistics.median(b[k])
        d = 100.0 * (mb - ma) / ma
        worst = max(worst, abs(d))
        print(f"| {k[0]} | {k[1]} | {k[2]} | {ma/1e6:.4f} | {mb/1e6:.4f} | {d:+.1f} |")
    print(f"\nworst |diff| between the two runs: {worst:.1f}%")

    # headline table: median ms and ns/elem, speedup vs V1 single-thread
    print(f"\n## headline table ({paths[0]}, median of 20)\n")
    base = {}
    for size in SIZES:
        base[size] = statistics.median(primary[("1-single", "V1", size)])
    print("| variant | N | median ms | ns/elem | GB/s moved | speedup vs V1 1T |")
    print("| --- | --: | --: | --: | --: | --: |")
    rows = [("V1", "1-single"), ("V1s", "1-single"), ("V3", "1-single"),
            ("V3u2", "1-single"), ("V3e", "1-single"), ("S1", "1-single"),
            ("S3", "1-single"), ("COPY", "1-single"),
            ("V2", "2-parallel"), ("V4", "2-parallel"), ("DISP", "2-parallel"),
            ("TCREATE", "3-percall"), ("V2p", "3-percall"), ("V4p", "3-percall")]
    for var, ph in rows:
        for size in SIZES:
            k = (ph, var, size)
            if k not in primary:
                continue
            s = stats(primary[k])
            m = s["med"]
            nspe = m / size
            gbs = size * 8 / m * 1e3 / 1e9 if False else size * 8 / (m * 1e-9) / 1e9
            print(f"| {var} | {size} | {m/1e6:.4f} | {nspe:.3f} | {gbs:.2f} | "
                  f"{base[size]/m:.2f}x |")

    # ratios measured inside one phase (adjacent, same conditions)
    print("\n## in-phase ratios\n")
    for size in SIZES:
        g = lambda ph, v: statistics.median(primary[(ph, v, size)])
        print(f"N={size}:")
        print(f"  V3/V1 (1T, phase 1)      = {g('1-single','V1')/g('1-single','V3'):.3f}x"
              f"   [{g('1-single','V1')/1e6:.4f} -> {g('1-single','V3')/1e6:.4f} ms]")
        print(f"  V3/V1 (1T, phase 2)      = {g('2-parallel','V1')/g('2-parallel','V3'):.3f}x")
        print(f"  V2/V1 4-thread scaling   = {g('2-parallel','V1')/g('2-parallel','V2'):.3f}x")
        print(f"  V4/V3 4-thread scaling   = {g('2-parallel','V3')/g('2-parallel','V4'):.3f}x")
        print(f"  V4/V1 combined           = {g('1-single','V1')/g('2-parallel','V4'):.3f}x")
        print(f"  V3u2/V1 (1T)             = {g('1-single','V1')/g('1-single','V3u2'):.3f}x")
        print(f"  V3/V3e divide share      = {(g('1-single','V3')-g('1-single','V3e'))/1e6:.4f} ms")
        print(f"  V4 vs COPY floor         = {g('2-parallel','V4')/g('1-single','COPY'):.2f}x above")
        print(f"  dispatch overhead        = {g('2-parallel','DISP')/1e6*1000:.2f} us"
              f" ({100*g('2-parallel','DISP')/g('2-parallel','V2'):.2f}% of V2)")

if __name__ == "__main__":
    main(sys.argv[1:])
