# Why a baked 64-bit constant call fails through `jit_loader` and a loaded pointer works

Scope: the measured asymmetry between

* **constant form** — `((int(*)(void*,void**,int))ADDR)(group, ctx, 4)` → callee returns nonzero, the
  serial fallback runs: correct output, no parallelism, no dispatch;
* **volatile-load form** — `volatile unsigned long t[2]; t[0]=ADDR>>32; t[1]=ADDR;
  unsigned long a=(t[0]<<32)|t[1]; fp=(rk_dispatch_t*)a; fp(...)` → returns 0, the workers write the
  output.

Both reported by the parent session as measured, in this tree, through `CPUProgram`/`jit_loader`, with
`link_sym('rk_run', DLL._loaded_.values()) == base + 0x8d8`.

Everything below is **read from source only**. No program, compiler, `readelf`, or NPU call was run for
this report, because a timing measurement was in flight. Claims that need an executed command are marked
**not established** and the exact command is given in §8.

---

## 0. Summary

1. A call to a **literal** 64-bit address needs **no relocation this loader is missing**. clang either
   builds the address with `movz`/`movk` (no relocation, no `.rodata`) or loads it from a `.rodata`
   constant pool via `adrp` + `ldr`, whose two relocations
   (`R_AARCH64_ADR_PREL_PG_HI21`, `R_AARCH64_LDST64_ABS_LO12_NC`) **are implemented and are correctly
   encoded** (`elf.py:71-77`). `.rodata` **is** copied into the image at a consistent offset
   (`elf.py:33,37-41`).
2. `R_AARCH64_ABS64` (257) is **not** handled — a real defect — but it fires only for a
   **symbol-valued** `.quad`, not for a literal address, and it raises `NotImplementedError`
   (`elf.py:84`) at load time. **A relocation the loader cannot apply cannot produce "loads, runs, and
   returns nonzero"; it produces a hard failure at program construction.** That distinction removes the
   relocation-gap hypothesis from the top of the list for the *measured* symptom.
3. The mechanisms that *can* produce "returns nonzero, no crash" are, in order: (a) `rk_run` was entered
   but **refused** because `rk_failed` is sticky in `librkpool.so`; (b) the address that was executed is
   **not the address `link_sym` returns in this process** (the baked address is process-specific and the
   disk cache is not); (c) the function the loader called at image offset 0 **is not the entry**.
4. The safest fix is **not** in `elf.py`: stop baking addresses. Route `rk_run` through the ordinary
   undefined-symbol path (`R_AARCH64_CALL26`, handled, and already proven bitwise-equal and 3.86x
   standalone in `docs/rknpu/results/splitcheck/`), or, as the immediate stopgap, keep the measured
   volatile round-trip. Adding `R_AARCH64_ABS64` is worth doing on its own merits but would not change
   this measurement unless the object actually contains type 257 (§2, §8).

---

## 1. Q1 — how does clang emit a call to a 64-bit constant address, and with what relocations?

### What the source is compiled with

`tinygrad/runtime/support/compiler_cpu.py:21-22`:

```python
    return subprocess.check_output([getenv("CC", 'clang'), '-c', '-x', 'c', '-O2', '-fPIC', '-ffreestanding', '-fno-math-errno', '-nostdlib',
                                    '-fno-ident', f'--target={self.arch}-none-unknown-elf', *self.args, '-', '-o', '-'], input=src.encode('utf-8'))
```

`*self.args` for arm64 is `-ffixed-x18 -mcpu=<native or explicit>` (`compiler_cpu.py:13`). The output is
an **ET_REL relocatable object** (`.o`), not a shared object. That matters: ET_REL has `sh_addr == 0` for
every allocated section, and it has no dynamic relocations at all.

`ClangRenderer` (`tinygrad/renderer/cstyle.py:284-292`) contributes nothing here — it only chooses types,
`__fp16`, `restrict`, and vector typedefs. The baked address is inserted into the **source text** by the
caller, not by the renderer:

`extra/rockchip/thread_split.py:99-102`
```python
  volatile unsigned long target[2];
  target[0] = {address}ULL >> 32; target[1] = {address}ULL & 0xFFFFFFFFUL;
  unsigned long addr = (target[0] << 32) | target[1];
  rk_dispatch_t *fp = (rk_dispatch_t *)addr;
```
and, for the constant form, `extra/rockchip/thread_split.py:103` `if (fp({name}_group, ctx, {threads}) != 0)`
is replaced in the variant under test by `((int (*)(rk_fn_t *, void **, int))ADDR)(...)` — the same shape
as `extra/rockchip/attn.c:92`:

```c
  if (((int (*)(rk_fn_t *, void **, int))RK_POOL_ADDR_ULL)(rockchip_attn_group, ctx, 4) != 0)
```

So the compiler sees a *literal integer* converted to a function pointer and called. `-fPIC` does not
change which of these emissions is legal, because an immediate address is position-independent
either way.

### The three possible emissions

| # | instruction sequence | sections | relocations in `.rela.text` | handled by `relocate`? |
| --- | --- | --- | --- | --- |
| A | `movz x8,#hi16` + up to 3 × `movk x8,#…,lsl` + `blr x8` | `.text` only | **none** (immediates are final) | n/a — nothing to relocate |
| B | `adrp x8, .LCPI` + `ldr x8,[x8,#:lo12:.LCPI]` + `blr x8` | `.text` + `.rodata` | `R_AARCH64_ADR_PREL_PG_HI21` (275) and `R_AARCH64_LDST64_ABS_LO12_NC` (286) | **yes** — 275 at `elf.py:71`, 286 at `elf.py:77` |
| C | `adrp x8, sym` + `add x8,x8,#:lo12:sym` + `blr x8` — the **address of a symbol** (e.g. the `&_group` argument), not how a literal value is obtained | `.text` only (target is intra-image) | 275 + `R_AARCH64_ADD_ABS_LO12_NC` (277) | **yes** — 277 at `elf.py:74` |

All three end in an indirect `blr`, never `R_AARCH64_CALL26`; a constant has no symbol to branch to, and
A64 has no absolute branch.

Two further possibilities exist only for a **symbol-valued** constant (e.g. `static const void *p = &rk_run;`,
or a jump table in `.rodata`), and neither is how a literal address is emitted:

| # | shape | relocation | handled? |
| --- | --- | --- | --- |
| D | `.quad &sym` in `.rodata` | `R_AARCH64_ABS64` (257) in `.rela.rodata` | **no** — `elf.py:84` raises |
| E | address taken through the GOT (non-`static`, `-fPIC`) | `R_AARCH64_ADR_GOT_PAGE` (311) + `R_AARCH64_LD64_GOT_LO12_NC` (312) | **no** — same raise; and no `.got` section is allocated |

**Which one this object uses: not established.** It is a property of clang's instruction selection for a
given literal, not of anything in this repository, and the object cannot be read as text. The one command
that settles it is in §8. The `-fPIC`-specific worry in the question — "the constant load needs ABS64" —
does **not** follow: `-fPIC` affects how *symbol addresses* are addressed, not how a known integer is
materialised, and a literal in `.rodata` is raw bytes with **no relocation in `.rela.rodata` at all**.

---

## 2. Q2 — does `elf_loader` place `.rodata` correctly, and does `relocate` cover the constant load?

### 2.1 `.rodata` placement: yes

`elf.py:33`:
```python
  progbits = [sh for sh in sections if sh.header.sh_type == libc.SHT_PROGBITS]
```
`elf.py:36-41`:
```python
  image = bytearray(max([sh.header.sh_addr + sh.header.sh_size for sh in progbits if sh.header.sh_addr != 0] + [0]))
  for sh in progbits:
    if sh.header.sh_addr != 0: image[sh.header.sh_addr:sh.header.sh_addr+sh.header.sh_size] = sh.content
    else:
      image += b'\0' * (((align:=max(sh.header.sh_addralign, force_section_align)) - len(image) % align) % align) + sh.content
      sh.header.sh_addr = len(image) - len(sh.content)
```

For an ET_REL object every `sh_addr` is 0, so sections are appended in section-header order with
`max(sh_addralign, force_section_align)` padding, and `sh_addr` is **rewritten in place** to the offset it
landed at. `.rodata` is `SHT_PROGBITS` (1, `libc.py:1521`), so it is placed, and because the rewrite happens
*before* the relocation pass (`elf.py:44-51`), every relocation against a `.rodata` symbol resolves to the
offset the bytes actually have. `.text` is the first `PROGBITS` in a clang object, so it lands at offset 0 —
which is what `self.fxn` depends on (`ops_cpu.py:46`).

Alignment is honoured as a floor (`max(...)`), so an 8- or 16-byte constant is correctly aligned.

**Counter-example in the same function — `.bss` is never placed.** `SHT_NOBITS` (8, `libc.py:1528`) is not
in `progbits`, so `.bss` gets no image slot, its `sh_addr` stays 0, and every symbol in it resolves through
`elf.py:50-51` to

```python
                sections[sym.st_shndx].header.sh_addr + sym.st_value,
```

i.e. **`0 + 0`**. This is the measured "file-scope data is at address 0" defect
(`mistake.md:963-967`, `docs/rknpu/results/threadsplit/README.md:39-51`). It is the one proven *silent*
wrong-address bug in this loader, and it matters for the ranking in §4: it shows the loader's failure mode
in this area is "plausible wrong address", not "crash".

### 2.2 The two relocations a constant load needs are implemented

`elf.py:71-78`:
```python
      case libc.R_AARCH64_ADR_PREL_PG_HI21:
        rel_pg = (tgt & ~0xFFF) - (ploc & ~0xFFF)
        return instr | (getbits(rel_pg, 12, 13) << 29) | (getbits(rel_pg, 14, 32) << 5)
      case libc.R_AARCH64_ADD_ABS_LO12_NC: return instr | (getbits(tgt, 0, 11) << 10)
      case libc.R_AARCH64_LDST16_ABS_LO12_NC: return instr | (getbits(tgt, 1, 11) << 10)
      case libc.R_AARCH64_LDST32_ABS_LO12_NC: return instr | (getbits(tgt, 2, 11) << 10)
      case libc.R_AARCH64_LDST64_ABS_LO12_NC: return instr | (getbits(tgt, 3, 11) << 10)
      case libc.R_AARCH64_LDST128_ABS_LO12_NC: return instr | (getbits(tgt, 4, 11) << 10)
```

Checked against the encodings, with `getbits(v,s,e) = (v >> s) & ((1 << (e-s+1)) - 1)` (`helpers.py:94`):

* `immlo` = bits 12..13 of the page delta → `<<29` (bits 30:29) ✓
* `immhi` = bits 14..32 (19 bits) → `<<5` (bits 23:5) ✓; negative deltas work because the mask carries the
  two's-complement bits
* `LDST64` imm12 uses bits 3..11 → scale 8 ✓; `LDST128` bits 4..11 → scale 16 ✓; `ADD` bits 0..11 ✓

`R_AARCH64_ADR_PREL_PG_HI21` is PC-relative, so the base cancels **provided the load address is
page-aligned**. It is: `HostAllocator._alloc` uses `mmap.mmap(-1, size, ...)` (`device.py:296`), and
`mv_address` is `ctypes.addressof(ctypes.c_char.from_buffer(mv))` (`helpers.py:576`); mmap is page-aligned.
This assumption would break for an `external_ptr` allocation (`device.py:294`) — a latent constraint, not
this bug.

Therefore, **by inspection, the `.rodata` route works**. If the failing object uses emission A, the loader
is not involved at all; if it uses B, the loader is involved but correct. Only D or E break, and both raise.

### 2.3 `R_AARCH64_ABS64` is not handled — a real defect

`libc.py:3031` defines `R_AARCH64_ABS64 = 257`. `elf.py:61-84` has no case for 257 (nor 258/259/261, nor
263-272, nor 276, nor 278, nor 300-313). The fallthrough is

`elf.py:84`:
```python
    raise NotImplementedError(f"Encountered unknown relocation type {r_type}")
```

So any object containing an `R_AARCH64_ABS64` — a `.quad` holding a *symbol* value, which is what
`-fPIC` data initialisers and `.rodata` jump tables produce — **fails to load**. That is a concrete defect
and it is worth fixing.

The fix is not a single `case`, because the write is hard-coded to 4 bytes:

`elf.py:87-88`:
```python
  for ploc,tgt,r_type,r_addend in relocs:
    image[ploc:ploc+4] = struct.pack("<I", relocate(struct.unpack("<I", image[ploc:ploc+4])[0], base, ploc, tgt+r_addend, r_type, r_addend))
```

A 64-bit relocation needs `image[ploc:ploc+8] = struct.pack('<Q', …)` and a read of 8 bytes. `ops_amd.py`
already does exactly that for its own type (`ops_amd.py:521-523`), so the pattern exists in-tree:

```python
  for off, sym, typ, addent in relocs:
    assert typ == 5, f"unknown AMD reloc {typ}"  # R_AMDGPU_REL64
    image[off:off+8] = struct.pack('<q', sym - off + addent)
```

**Critical for this investigation:** adding 257 would *not* change the measured failure, because 257 only
appears when the `.quad` holds a symbol, and when it does the loader raises **loudly at load time** — the
kernel never runs, so it cannot "return nonzero". The same is true of the MOVW family (264/266/268/269),
`ADR_PREL_PG_HI21_NC` (276), and the GOT pair (311/312).

---

## 3. Q3 — `R_AARCH64_RELATIVE` and other dynamic relocations

`link_sym` (`elf.py:9-14`) resolves only **undefined symbol names**:

```python
def link_sym(sym:str, libs:list[ctypes.CDLL], link_syms:dict[str,int]|None=None) -> int:
  if link_syms is not None and (addr:=link_syms.get(sym)): return addr
  for lib in libs:
    try: return unwrap(ctypes.cast(getattr(lib, sym), ctypes.c_void_p).value)
    except (OSError, AttributeError, TypeError): pass
  raise RuntimeError(f'Attempting to relocate against an undefined symbol {sym}')
```

`relocate` handles **no** dynamic relocation: `R_AARCH64_RELATIVE` (1027, `libc.py:3147`) and
`R_AARCH64_GLOB_DAT` (1025, `libc.py:3145`) are defined in the autogen bindings and never referenced in
`elf.py`.

For this path that is *correct*, not a gap: `jit_loader` is only ever given the output of `clang -c`, an
ET_REL object, and a relocatable object cannot contain `RELATIVE`/`GLOB_DAT`/`JUMP_SLOT` — those are
produced by a link step that has not happened. A constant-pool entry holding a **literal** needs no
relocation at all; holding a **symbol** it needs the *static* `R_AARCH64_ABS64`, which is missing (§2.3).
`elf_loader` is shared with backends that do see ET_DYN/HSACO-style blobs, but they apply their own
relocation logic (`ops_amd.py:519-523`) or collect them for their own patching pass (`ops_nv.py:261-264`).

`.got` is also never allocated: it is `SHT_PROGBITS`, so it *would* be appended like any other section, but
nothing resolves `R_AARCH64_ADR_GOT_PAGE`/`LD64_GOT_LO12_NC`, and a `.o` has no `.got` contents to place.
GOT-based access is therefore unsupported by construction, and fails loudly.

---

## 4. Q4 — why does the CALL26 path survive while the constant path does not?

### 4.1 What the CALL26 path actually does

`elf.py:79-83`:
```python
      case libc.R_AARCH64_CALL26 | libc.R_AARCH64_JUMP26:
        if -(2**25) <= tgt-ploc-base and tgt-ploc-base <= (2**25 - 1) * 4: return instr | getbits(tgt-ploc-base, 2, 27)
        # create trampoline:         LDR x17, 8  BR x17
        image += bytes(-len(image) % 4) + struct.pack("<IIQ", 0x58000051, 0xD61F0220, tgt)
        return instr | getbits(len(image)-ploc-16, 2, 27)
```

Three properties differ from the constant path, and all three are **loud**:

1. **`base` is subtracted** (`tgt-ploc-base`), and `base` is the real load address (`ops_cpu.py:31`
   passes `self.buf.buf`). The branch offset is exact regardless of where the image lands.
2. **Range is checked** and out-of-range targets get a real trampoline (`LDR x17,#8; BR x17; .quad tgt`).
3. **Every relocation the symbol path needs is implemented** (`CALL26`, and for the callee *address* —
   `&_group` — `ADR_PREL_PG_HI21` + `ADD_ABS_LO12_NC`, both intra-image and both correct).

The constant path has no equivalent guarantee: if the emitted form is one the loader does not implement, it
raises; if it is one it implements, the value is right. There is **no silent middle ground in
`relocate`** — which is why a relocation gap is a poor explanation for the measurement.

### 4.2 Ranked mechanisms, by how well they explain "returns nonzero, no crash"

| # | mechanism | silent? | fits "nonzero, no crash"? | evidence in source | discriminating check |
| --- | --- | --- | --- | --- | --- |
| 1 | `rk_run` **was entered and refused**: `rk_failed` is sticky for the process lifetime, and -1 sends the wrapper to the serial loop | yes | **exactly** — nonzero, fallback, correct output, no workers | `rkpool.c:84-89` (`if (!rk_started) { … if (pthread_create(…) != 0) rk_failed = 1; } if (rk_failed) { rk_refused_total++; return -1; }`) | print `rk_refused()`; it is the only counter that is actually incremented (§7.3) |
| 2 | the executed constant is **not** `link_sym`'s value in this process (stale / foreign baked address) | yes | yes — lands in mapped text, returns garbage nonzero; the *symbol* form is immune because it is resolved at load | `thread_split.py:66-69`: "it changes per process, so the source (and hence the disk cache key) changes too"; `renderer/__init__.py:15-19` + `helpers.py:426-431` cache on `src` only | kernel writes back the `unsigned long` it called **and** `link_sym('rk_run', …)` into the output buffer, in the same process |
| 3 | the loader's entry is **not the entry**: `self.fxn` always jumps to image offset 0 | yes | yes — `_part` returns `void`, x0 is garbage ⇒ "nonzero" ⇒ fallback, no crash | `ops_cpu.py:46`: `self.fxn = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(self.buf.buf)`; the failure is measured in `thread_split.py:88-90` | `readelf -s obj`: entry must be `.text`, `st_value == 0` |
| 4 | a relocation the loader cannot apply (257/264-269/276/311/312) | **no** — raises | no | `elf.py:84` | `readelf -r obj` |
| 5 | `.rodata` placement or ADRP/LDST encoding wrong | would be silent | only if the wrong value happens to be callable | §2.1-2.2: placement and encodings check out; base is page-aligned | kernel writes back the value it loaded |
| 6 | CALL26 trampoline overflows the allocation | **no** — raises | no | `ops_cpu.py:30-32` allocates `len(obj.lib)`, then `host[:len(loaded)] = loaded`; `MMIOInterface.__setitem__` forwards to a `memoryview` of exactly `nbytes` (`memory.py:7-10`) | always exercised once a trampoline is emitted (§7.2) |

Mechanism 1 outranks 2 only because it requires no loader or codegen defect at all and reproduces the
symptom exactly; if the two forms were measured **in the same process at the same time**, `rk_failed` would
have to be 0 for both and 1 drops below 2. That fact is not established here.

Mechanism 4 is listed because it is the concrete defect in the question, but it is *fourth* on explanatory
power: `NotImplementedError` at `ops_cpu.py:31` aborts program construction, and the parent observed a
kernel that ran and returned nonzero.

Mechanism 3 explains everything about the *earlier* invalid threadsplit numbers and is the reason this
repo insists the entry be emitted first (`thread_split.py:88-90`); it does not by itself distinguish the
constant form from the volatile form, since both wrappers put the entry first.

### 4.3 Why the two source forms can differ at all

This part of the question's model is sound. An immediate forces the compiler to keep the value in
registers or in a read-only pool; `volatile` forces a store **and** a reload, so the value passes through
the stack and no `.rodata` entry is needed:

`thread_split.py:96-98`
```c
  /* The target must be loaded from memory. A materialised constant call never reaches the
     pool through jit_loader (measured: returns nonzero, fallback runs, no speedup), while this
     volatile round-trip does (measured: rc=0 and the workers write the output). */
```

So the forms *can* differ in whether `.rodata`/`.rela.text` entries exist. What they cannot differ in,
given the code above, is the **value** — unless the address itself is wrong (§4.2 #2), which is exactly why
the write-back probe in §8 is the decisive one.

---

## 5. Q5 — where the fix belongs

### Option A — emission: always load the target through a `volatile` local (measured to work)

* **Where:** `extra/rockchip/thread_split.py:96-104` (already there) and `extra/rockchip/attn.c:88-95`
  (bakes directly at line 92, does **not** use the round-trip).
* **Cost:** two stores and two loads per kernel call, i.e. nanoseconds against 0.4-7 ms kernels.
* **Pro:** measured working today; zero risk; no loader change; unblocks an in-model A/B immediately.
* **Con:** it is a workaround for an undiagnosed defect. It stays correct only while the compiler keeps the
  spill; nothing in the source forbids re-materialising a constant. It does not fix `.bss` at 0, the
  trampoline overflow, or `ABS64`. And the address stays baked, so the source — and the disk cache key —
  stays **process-specific** (`thread_split.py:66-69`), which means a cache hit from another process is a
  jump to a meaningless address. Today that is masked only because the address string changes every run.

### Option B — `elf.py`: add the missing relocation cases

* **Where:** `elf.py:61-84` (the `match`), plus the write loop `elf.py:87-88` for 8-byte types.
* **Minimal version:** `case libc.R_AARCH64_ABS64: return i2u(64, tgt)` with a width-aware write, e.g.
  `w = 8 if r_type == libc.R_AARCH64_ABS64 else 4`.
* **Pro:** closes a real hole; any `.quad &sym` object currently cannot load at all; it is the only place
  that can ever make GOT-qualified or MOVW forms work.
* **Con:** **it would not have fixed the measured case** unless `readelf -r` shows one of those types
  (§2.3, §8). Fixing a relocation the object does not contain is a change with no observable effect — the
  exact failure mode `AGENTS.md` warns about ("a flag that silently does nothing makes an experiment a
  duplicate of its own control"). GOT types cannot be fixed here at all without allocating a `.got`.
  This is a correctness improvement, not a fix for this symptom.

### Option C — `ops_cpu.py`

* Nothing in `_load`/`__init__` can explain the asymmetry: the image is built, copied, `mprotect`ed and
  entered identically for both forms (`ops_cpu.py:30-35,46`).
* Two genuine defects do live here and should be fixed independently:
  * **allocation/trampoline mismatch** — `dev.allocator.alloc(len(obj.lib), …)` then
    `host[:len(loaded)] = loaded`; `jit_loader` can return **more** bytes than `len(obj.lib)` because it
    appends 16-byte trampolines (`elf.py:67,82`). The `memoryview` slice assignment is bounds-checked
    (`memory.py:10`), so any kernel that needs a trampoline fails with `ValueError` instead of loading.
    Allocate `len(obj.lib) + 64`, or `_load` first and allocate `len(loaded)`.
  * **`mprotect` return value ignored** (`ops_cpu.py:35`), and the `__clear_cache`/`msync` path is chosen by
    "is `rt` loaded" (`ops_cpu.py:41-44`) — `msync` is not an I-cache flush.

### Recommendation

1. **Do not change `elf.py` for this bug yet.** Run the `readelf -r` check (§8). If it shows 275/277/286
   only, the loader is not involved and a relocation patch is dead code.
2. **Stop baking the address.** Switch `attn.c:92` and the `thread_split` wrapper to the **symbol** form:
   `extern int rk_run(rk_fn_t *, void **, int);` called directly, which needs `R_AARCH64_CALL26` — handled,
   base-corrected, with a range check — and resolves through `DLL._loaded_` at load time. The symbol form
   is already the one with a working, bitwise-exact positive control: `docs/rknpu/results/splitcheck/README.md:10-11`
   reports `differing=0 verdict=BITWISE-EQUAL` and `3.86x`. It also makes the object **process-independent**,
   so the disk cache key stops encoding an address (`renderer/__init__.py:15-19`).
   Preconditions, both already known in-tree: register `librkpool.so` in `DLL._loaded_` before the kernel
   loads (`thread_split.py:181-185`, because `link_libs=list(DLL._loaded_.values())`, `ops_cpu.py:19-20`),
   and fix the trampoline overflow (Option C) because a `.so` mapping is often >128 MB from the image.
3. **Until (2) is measured, keep the volatile round-trip** — it is the only form measured to work, and it
   is already committed in `thread_split.py`.

---

## 6. What this does not establish

* **Which emission clang actually produced** for the failing object — A or B for a literal, D if the
  constant is symbol-valued. No `readelf` was run. Everything in §2 is "the loader handles A and B
  correctly", not "the object contains them".
* **Whether the two forms were measured in the same process.** That decides rank 1 vs rank 2 in §4.2.
* **Whether `rk_refused()` was nonzero** in the failing run. It is the single most informative number and it
  was not reported.
* **Whether the failing object came from `compile_cached` or from a direct `jit_loader` call.** If from the
  cache, the address in the executed object may predate the process it ran in.

---

## 7. Independent defects found while reading (all read-only)

### 7.1 `.bss` is never allocated, so every `.bss` symbol is at address 0

`elf.py:33` filters `SHT_PROGBITS` only; `SHT_NOBITS` (8) is excluded. Combined with `elf.py:50-51`, a
`.bss` symbol resolves to `0 + st_value`, and for the common `static int g_x;` that is `0`. This is the
measured `&g_x == 0x0` of `mistake.md:963-967`. It is the same *class* as the constant-address question
(a section the loader does not place), and it is silent — which is why it cost two runs.

### 7.2 A CALL26 trampoline cannot fit in the buffer

`ops_cpu.py:30-32` allocates exactly `len(obj.lib)` bytes and then assigns `host[:len(loaded)] = loaded`,
while `elf.py:67,82` append 16/14-byte trampolines to the image. `MMIOInterface.__setitem__` forwards to a
`memoryview` of exactly `nbytes` (`memory.py:7-10`), so the assignment raises `ValueError` rather than
truncating. This is latent today only because the observed CALL26 targets happened to be in range.

### 7.3 `rk_calls_total` is never incremented — "dispatch calls 0" is not evidence

`extra/rockchip/rkpool.c:47-50` declares and reads it; the only counter incremented anywhere is
`rk_refused_total` at line 89. `rk_calls_total++` appears nowhere in the file. Therefore the line

    [rkpool] dispatch calls 0  refused/serial 0

(`docs/rknpu/results/splitcheck/README.md:20`, and `threadsplit/attrib14.log:59`, `attrib22.log:64`,
`attrib17.log:64`) **cannot ever print a nonzero first number**, and the two READMEs that cite it as proof
that "the pool was never reached" (`splitcheck/README.md:20`, `threadsplit/README.md:5`) are resting on a
counter with no writes. The `refused/serial` field *is* meaningful, and it was 0 — i.e. `rk_run` never
returned -1 in those runs, which is consistent both with "never entered" and with "entered and succeeded".
Adding `rk_calls_total++` at the top of `rk_run` is a one-line fix that makes all future runs interpretable.

### 7.4 `.eh_frame` relocations are skipped deliberately

`elf.py:46` `if trgt_sh_name == ".eh_frame": continue` — fine for execution (unwinding only), but it means
`image` contains `.eh_frame` with unpatched PC-relative data; backtraces through a jit-loaded kernel will
be wrong. Not related to this bug; recorded so it is not rediscovered.

---

## 8. The checks that settle it (run only when the machine is idle)

```
# 1. What the call site actually needs. This answers Q1 for this object.
readelf -rW <object.o> | grep -E 'R_AARCH64'
readelf -SW <object.o> | grep -E '\.(text|rodata|data|bss)'      # confirm .rodata exists at all

# 2. Is the entry at image offset 0? (mechanism 3)
readelf -sW <object.o> | grep -E ' (T|t) '                        # entry must be .text, st_value 0

# 3. Split the hypothesis space with one probe kernel, no timing involved:
#    write back (a) the 64-bit value actually used for the call and (b) link_sym('rk_run', DLL._loaded_.values())
#    into two output words. Equal => the address was right and the difference is inside rk_run (rank 1).
#    Different => the address was wrong; compare its 8 bytes in the loaded image against .rodata on disk
#    to separate "loader placed it wrong" (rank 5) from "stale address" (rank 2).

# 4. rk_refused() at process end, for both forms, in the same process:
python -c "..."  # print lib.rk_refused() after each variant, alongside lib.rk_calls() once 7.3 is fixed
```

If (1) shows only 275/277/286 the loader is exonerated for this symptom and the answer is §4.2 rank 1 or 2.
If it shows 257, 264-269, 276, 311 or 312, then either the object could not have loaded at all (the loader
raises at `elf.py:84`) or the harness swallowed that exception — in both cases the object being reasoned
about is not the object that ran, and the measurement must be redone against the object actually loaded.
