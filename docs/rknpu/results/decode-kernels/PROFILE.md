# Host profile of the decode path (py-spy)

`py-spy record -f speedscope --rate 200` over 60 decode steps: 24252 samples, weight 121.3. `flameox` is unavailable on this host (rea-agents 6.7.0 predates it;
6.8+ needs Node >= 24.22.1, which would repoint nvm away from the 24.12.0 that dsh/dst need), and the
REA MCP server is not connected in this session, so py-spy is the profiler in use.


## Whole process: weight 118.4 of 121.3 (97.7%)

| frame (self time) | share | file |
| --- | --: | --- |
| `CPUProgram.__call__` | 43.6% | `tinygrad/runtime/ops_cpu.py` |
| `Compiled.run` | 9.1% | `tinygrad/llm/kernels/replay_c.py` |
| `MMIOInterface.__setitem__` | 1.9% | `tinygrad/runtime/support/memory.py` |
| `FileIOInterface.ioctl` | 1.8% | `tinygrad/runtime/support/system.py` |
| `_apply_map_to_tensors.<locals>.visitor` | 1.4% | `tinygrad/tensor.py` |
| `recursive_property.__get__` | 1.3% | `tinygrad/uop/ops.py` |
| `DLL.bind.<locals>.wrap.<locals>.wrapper` | 1.0% | `tinygrad/runtime/support/c.py` |
| `UOp.topovisit` | 0.9% | `tinygrad/uop/ops.py` |

## Decode phase (`Compiled.run` in stack): weight 11.1 of 121.3 (9.1%)

| frame (self time) | share | file |
| --- | --: | --- |
| `Compiled.run` | 97.1% | `tinygrad/llm/kernels/replay_c.py` |
| `UOp.buffer` | 0.1% | `tinygrad/uop/ops.py` |
| `Compiled.run` | 0.1% | `tinygrad/llm/kernels/replay_c.py` |
| `UOpMetaClass.__call__` | 0.1% | `tinygrad/uop/ops.py` |
| `UOp.is_variable` | 0.1% | `tinygrad/uop/ops.py` |
| `compiled_match` | 0.1% | `<string>` |
| `UOp.toposort` | 0.1% | `tinygrad/uop/ops.py` |
| `to_mv` | 0.1% | `tinygrad/helpers.py` |

The whole-process table is dominated by weight packing at model load; only the filtered table
describes decode. Reading the unfiltered top frame as a decode cost is the trap this file records.
