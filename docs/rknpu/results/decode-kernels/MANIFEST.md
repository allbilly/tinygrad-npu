# Decode kernels: source and in-model timing

One warm decode token, `MAX_CALLS_PER_SEG = 1`. Wall **182.28 ms**, 881 calls, 33 distinct kernels.

| kernel | class | calls | ms | us/call | submit ms | cpu ms | source |
| --- | --- | --: | --: | --: | --: | --: | --- |
| `rockchip_linear_1_9216_2560` | fused INT8 Linear | 64 | 49.83 | 778.6 | 48.70 | 1.13 | `rockchip_linear_1_9216_2560.c` |
| `rockchip_linear_1_2560_4096` | fused INT8 Linear | 96 | 40.03 | 417.0 | 38.63 | 1.40 | `rockchip_linear_1_2560_4096.c` |
| `rockchip_linear_1_24576_2560` | fused INT8 Linear | 10 | 20.61 | 2061.3 | 20.04 | 0.57 | `rockchip_linear_1_24576_2560.c` |
| `rockchip_linear_1_8192_2560` | fused INT8 Linear | 24 | 16.69 | 695.2 | 16.17 | 0.52 | `rockchip_linear_1_8192_2560.c` |
| `rockchip_linear_1_4096_2560` | fused INT8 Linear | 24 | 8.66 | 361.0 | 8.31 | 0.36 | `rockchip_linear_1_4096_2560.c` |
| `rockchip_linear_1_10240_2560` | fused INT8 Linear | 8 | 6.83 | 853.3 | 6.67 | 0.15 | `rockchip_linear_1_10240_2560.c` |
| `rockchip_scan_simd_1_32_1_128_128_1` | GDN scan | 24 | 5.91 | 246.2 | 0.00 | 5.91 | `rockchip_scan_simd_1_32_1_128_128_1.c` |
| `rockchip_linear_1_2560_1024` | fused INT8 Linear | 32 | 3.60 | 112.6 | 3.26 | 0.35 | `rockchip_linear_1_2560_1024.c` |
| `E_62080_4` | elementwise | 1 | 2.62 | 2616.8 | 0.00 | 2.62 | `E_62080_4.c` |
| `rockchip_attn_1_160` | attention | 8 | 2.30 | 287.3 | 0.00 | 2.30 | `rockchip_attn_1_160.c` |
| `E_2_16_32_4` | elementwise | 48 | 2.11 | 44.0 | 0.00 | 2.11 | `E_2_16_32_4.c` |
| `E_80_32_64_4` | elementwise | 1 | 1.59 | 1587.5 | 0.00 | 1.59 | `E_80_32_64_4.c` |
| `E_1024_4` | elementwise | 72 | 1.35 | 18.7 | 0.00 | 1.35 | `E_1024_4.c` |
| `r_16_32_4` | reduce | 48 | 1.24 | 25.8 | 0.00 | 1.24 | `r_16_32_4.c` |
| `E_8192_4` | elementwise | 24 | 1.18 | 49.4 | 0.00 | 1.18 | `E_8192_4.c` |
| `E_3072_3` | elementwise | 32 | 1.18 | 36.8 | 0.00 | 1.18 | `E_3072_3.c` |
| `E_8_32_4_4` | elementwise | 24 | 1.16 | 48.2 | 0.00 | 1.16 | `E_8_32_4_4.c` |
| `rockchip_linear_1_64_2560` | fused INT8 Linear | 24 | 0.97 | 40.4 | 0.65 | 0.32 | `rockchip_linear_1_64_2560.c` |
| `E_81920_4` | fill (kernel) | 8 | 0.69 | 86.1 | 0.00 | 0.69 | `E_81920_4.c` |
| `E_640_4` | elementwise | 97 | 0.63 | 6.5 | 0.00 | 0.63 | `E_640_4.c` |
| `r_640_4` | reduce | 65 | 0.45 | 6.9 | 0.00 | 0.45 | `r_640_4.c` |
| `E_16_64_4` | elementwise | 8 | 0.43 | 53.5 | 0.00 | 0.43 | `E_16_64_4.c` |
| `rockchip_argmax_1_248320` | reduce | 1 | 0.36 | 359.0 | 0.00 | 0.36 | `rockchip_argmax_1_248320.c` |
| `r_640_4_64_4` | reduce | 1 | 0.36 | 355.5 | 0.00 | 0.36 | `r_640_4_64_4.c` |
| `E_8_4` | elementwise | 48 | 0.32 | 6.6 | 0.00 | 0.32 | `E_8_4.c` |
| `rockchip_linear_1_2560_2560` | fused INT8 Linear | 1 | 0.29 | 285.2 | 0.23 | 0.05 | `rockchip_linear_1_2560_2560.c` |
| `E_8192_3` | elementwise | 24 | 0.22 | 9.2 | 0.00 | 0.22 | `E_8192_3.c` |
| `r_32_32_4` | reduce | 24 | 0.20 | 8.2 | 0.00 | 0.20 | `r_32_32_4.c` |
| `E_4_32_2_4_2` | elementwise | 8 | 0.07 | 9.0 | 0.00 | 0.07 | `E_4_32_2_4_2.c` |
| `r_16_64_4` | reduce | 8 | 0.07 | 8.5 | 0.00 | 0.07 | `r_16_64_4.c` |
| `E_16_32_2_4` | elementwise | 8 | 0.06 | 7.7 | 0.00 | 0.06 | `E_16_32_2_4.c` |
| `r_4_64_4` | reduce | 8 | 0.04 | 5.5 | 0.00 | 0.04 | `r_4_64_4.c` |
| `E` | elementwise | 8 | 0.04 | 4.4 | 0.00 | 0.04 | `E.c` |

## Class rollup

| class | ms | share |
| --- | --: | --: |
| fused INT8 Linear | 147.51 | 85.7% |
| elementwise | 12.94 | 7.5% |
| GDN scan | 5.91 | 3.4% |
| reduce | 2.71 | 1.6% |
| attention | 2.30 | 1.3% |
| fill (kernel) | 0.69 | 0.4% |

sum of kernels 172.07 ms; wall 182.28 ms; outside 10.22 ms
