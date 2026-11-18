#!/usr/bin/env python3
"""Emit the markdown tables pasted into report.md.

usage: ./gen_tables.py run-0.log > tables.md
"""
import sys, re, statistics, collections

SIZES = [131072, 1179648]

def load(path):
    cur = "?"; d = collections.defaultdict(list); seen = []
    for line in open(path):
        line = line.rstrip("\n")
        m = re.match(r"^# phase (\S+) n=(\d+)", line)
        if m:
            cur = m.group(1); continue
        if line.startswith("#") or not line:
            continue
        p = line.split(",")
        if len(p) != 6:
            continue
        key = (cur, p[0], int(p[1]))
        if key not in d:
            seen.append(key)
        d[key].append(float(p[3]))
    return d, seen

def med(d, ph, var, size):
    return statistics.median(d[(ph, var, size)])

def main(path):
    d, seen = load(path)

    print("### Summary\n")
    print("| variant | N | median ms | ns/element | GB/s moved | speedup vs V1 1 thread |")
    print("| --- | --: | --: | --: | --: | --: |")
    rows = [("V1", "1-single", "guarded scalar (emitted C shape)"),
            ("V1s", "1-single", "guarded scalar, stride 1"),
            ("V3", "1-single", "reference vector, 1 guard / 4 lanes"),
            ("V3u2", "1-single", "vector, 2 chains, 1 guard / 8 lanes"),
            ("V3e", "1-single", "vector exp2, divide removed"),
            ("S1", "1-single", "sigmoid, guarded scalar"),
            ("S3", "1-single", "sigmoid, reference vector"),
            ("COPY", "1-single", "y = x (memory floor)"),
            ("V2", "2-parallel", "V1 on 4 threads"),
            ("V4", "2-parallel", "V3 on 4 threads"),
            ("DISP", "2-parallel", "4-thread dispatch, no work"),
            ("TCREATE", "3-percall", "3 threads created + joined, no work"),
            ("V2p", "3-percall", "V1, 3 threads created per call"),
            ("V4p", "3-percall", "V3, 3 threads created per call")]
    for var, ph, _ in rows:
        for size in SIZES:
            k = (ph, var, size)
            if k not in d:
                continue
            m = statistics.median(d[k])
            base = med(d, "1-single", "V1", size)
            if var in ("DISP", "TCREATE"):
                print(f"| {var} | {size} | {m/1e6:.4f} | — | — | — |")
            else:
                print(f"| {var} | {size} | {m/1e6:.4f} | {m/size:.3f} | "
                      f"{size*8/m:.2f} | {base/m:.2f}x |")

    print("\n### 4-thread scaling, measured in phase 2 (adjacent runs)\n")
    print("| comparison | N=131072 | N=1179648 |")
    print("| --- | --: | --: |")
    def g(ph, v, size):
        return statistics.median(d[(ph, v, size)])
    def row(label, fn, fmt):
        cells = " | ".join(fmt(fn(s)) for s in SIZES)
        print("| " + label + " | " + cells + " |")
    row("V2 / V1 (scalar, 4 threads)", lambda s: g("2-parallel","V1",s)/g("2-parallel","V2",s), lambda x: "%.3fx" % x)
    row("V4 / V3 (vector, 4 threads)", lambda s: g("2-parallel","V3",s)/g("2-parallel","V4",s), lambda x: "%.3fx" % x)
    row("V4 / V1 (vector + 4 threads)", lambda s: g("1-single","V1",s)/g("2-parallel","V4",s), lambda x: "%.3fx" % x)
    row("V3 / V1 (vector, 1 thread)", lambda s: g("1-single","V1",s)/g("1-single","V3",s), lambda x: "%.3fx" % x)
    row("V3u2 / V1 (2 chains, 1 thread)", lambda s: g("1-single","V1",s)/g("1-single","V3u2",s), lambda x: "%.3fx" % x)
    row("V3e / V1 (divide removed, 1 thread)", lambda s: g("1-single","V1",s)/g("1-single","V3e",s), lambda x: "%.3fx" % x)
    row("V4 / COPY (how far above the copy floor)", lambda s: g("2-parallel","V4",s)/g("1-single","COPY",s), lambda x: "%.2fx" % x)
    row("dispatch overhead (DISP)", lambda s: g("2-parallel","DISP",s), lambda x: "%.2f us" % (x/1e3))

    print("\n### Raw values per repetition (ms, one kernel call per rep)\n")
    phases = [("1-single", ["V1", "V1s", "V3", "V3u2", "S1", "S3", "V3e", "COPY"]),
              ("2-parallel", ["V1", "V2", "V3", "V4", "DISP"]),
              ("3-percall", ["TCREATE", "V2p", "V4p"])]
    for size in SIZES:
        for ph, vars_ in phases:
            have = [v for v in vars_ if (ph, v, size) in d]
            n = len(d[(ph, have[0], size)])
            print(f"\n{ph}, N={size}, {n} reps\n")
            print("| rep | " + " | ".join(have) + " |")
            print("| --: | " + " | ".join("--:" for _ in have) + " |")
            for r in range(n):
                vals = []
                for v in have:
                    x = d[(ph, v, size)][r] / 1e6
                    vals.append(f"{x:.4f}" if x >= 0.001 else f"{x:.6f}")
                print(f"| {r} | " + " | ".join(vals) + " |")

if __name__ == "__main__":
    main(sys.argv[1])
