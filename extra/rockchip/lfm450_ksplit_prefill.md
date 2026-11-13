# Exact K-split down projection for LFM2.5-VL 450M

The guarded static-128 SwiGLU capability now covers both 896→4864→896 and
1024→4608→1024. Partial/symbolic chunks, scalar decode and other geometries
retain their existing path. The exact LFM/context-512 app profile enables the qualified K-split/packer
flags; other app profiles keep their previous recipe.

LFM retains the original K4096 activation maximum and scale. Two K2048 products
use exact slices of the original packed weight codes; their INT32 results are
added before FP32 conversion and dequantization. Quantizing each half separately
or adding dequantized floats would change the numerical contract.

| Segment | Header offset | Payload offset | Payload |
|---|---:|---:|---|
| First K2048 half | 0 | 512 | 128×2048 |
| Second K2048 half | 262656 | 263168 | 128×2048 |
| Original K512 tail | 525312 | 525824 | 128×512 |

The complete packet is 591360 bytes; the first two headers contain identical
scale bytes. The prepared halves add 64 MiB across 16 FFNs. Original weights
remain available for decode and fallback. These are immutable prepared inference
artifacts; changing weights requires reconfiguration.

Existing ordinary INT8 command geometry executes two M128/K2048 calls instead
of two M64/K4096 calls. The primitive passes fourteen full INT32 oracles, all
output guards, weight-slice checks and 29 interleaved FP16 health checks. Median
device time falls from 0.806 to 0.590 ms, with twelve positive paired blocks.
That timer excludes CPU activation packing, reduction and dequantization.

Four private full-model processes run control/split/split/control, each with
three warmups, three measured 128-input/64-output/context-512 requests and
200 ms idle. Control TTFT medians are 127.45/128.84 ms; split medians are
103.33/104.52 ms. All preserve the saved 32 full-vocabulary vectors and all
1,024 natural target vector hashes. The candidate includes the qualified batched
CPU producer as well as K-split; the full-model gain is not attributable solely
to NPU tiling. The generalized backend port independently passes those same
numerical gates and confirms all 32 intended K2048 calls.

The actual API gains10.65%in a balanced resident-worker test and passes two
fresh production workers. Current results and every slower/failed cohort are in
`~/oNPU/docs/benchmark-lfm450-ksplit-20261103.json`.
No sudo, clock/governor, IRQ or idle setting changes are needed. Original/RKLLM
quality, model maximum, image processing and concurrent generation remain open.
