# Decode kernels: source and in-model timing

One warm decode token, `MAX_CALLS_PER_SEG = 1`. Wall **196.18 ms**, 958 calls, 36 distinct kernels.

| kernel | class | calls | ms | us/call | submit ms | cpu ms | source |
| --- | --- | --: | --: | --: | --: | --: | --- |
| `rockchip_linear_1_9216_2560` | fused INT8 Linear | 64 | 49.84 | 778.8 | 48.68 | 1.17 | `rockchip_linear_1_9216_2560.c` |
| `rockchip_linear_1_2560_4096` | fused INT8 Linear | 96 | 40.04 | 417.0 | 38.61 | 1.43 | `rockchip_linear_1_2560_4096.c` |
| `rockchip_linear_1_24576_2560` | fused INT8 Linear | 10 | 20.13 | 2013.1 | 19.85 | 0.28 | `rockchip_linear_1_24576_2560.c` |
| `rockchip_linear_1_8192_2560` | fused INT8 Linear | 24 | 16.60 | 691.7 | 16.13 | 0.47 | `rockchip_linear_1_8192_2560.c` |
| `E_131072_4` | fill (kernel) | 48 | 8.76 | 182.6 | 0.00 | 8.76 | `E_131072_4.c` |
| `rockchip_linear_1_4096_2560` | fused INT8 Linear | 24 | 8.62 | 359.2 | 8.29 | 0.33 | `rockchip_linear_1_4096_2560.c` |
| `E_62080_4` | elementwise | 1 | 8.50 | 8497.3 | 0.00 | 8.50 | `E_62080_4.c` |
| `rockchip_linear_1_10240_2560` | fused INT8 Linear | 8 | 6.84 | 854.9 | 6.68 | 0.16 | `rockchip_linear_1_10240_2560.c` |
| `rockchip_scan_simd_1_32_1_128_128_1` | GDN scan | 24 | 5.22 | 217.4 | 0.00 | 5.22 | `rockchip_scan_simd_1_32_1_128_128_1.c` |
| `rockchip_linear_1_2560_1024` | fused INT8 Linear | 32 | 3.62 | 113.1 | 3.26 | 0.36 | `rockchip_linear_1_2560_1024.c` |
| `rockchip_attn_1_160` | attention | 8 | 2.22 | 277.7 | 0.00 | 2.22 | `rockchip_attn_1_160.c` |
| `E_124160_2` | elementwise | 1 | 2.09 | 2094.4 | 0.00 | 2.09 | `E_124160_2.c` |
| `E_2_16_32_4` | elementwise | 48 | 1.93 | 40.1 | 0.00 | 1.93 | `E_2_16_32_4.c` |
| `E_80_32_64_4` | elementwise | 1 | 1.58 | 1584.0 | 0.00 | 1.58 | `E_80_32_64_4.c` |
| `rockchip_linear_1_32_2560` | fused INT8 Linear | 48 | 1.45 | 30.2 | 0.94 | 0.52 | `rockchip_linear_1_32_2560.c` |
| `E_1024_4` | elementwise | 72 | 1.24 | 17.3 | 0.00 | 1.24 | `E_1024_4.c` |
| `E_3072_3` | elementwise | 32 | 1.18 | 36.9 | 0.00 | 1.18 | `E_3072_3.c` |
| `r_16_32_4` | reduce | 48 | 1.13 | 23.6 | 0.00 | 1.13 | `r_16_32_4.c` |
| `E_8_32_4_4` | elementwise | 24 | 1.10 | 46.0 | 0.00 | 1.10 | `E_8_32_4_4.c` |
| `E_8192_4` | elementwise | 24 | 0.66 | 27.7 | 0.00 | 0.66 | `E_8192_4.c` |
| `E_81920_4` | fill (kernel) | 8 | 0.66 | 82.9 | 0.00 | 0.66 | `E_81920_4.c` |
| `E_640_4` | elementwise | 97 | 0.64 | 6.6 | 0.00 | 0.64 | `E_640_4.c` |
| `r_640_4` | reduce | 65 | 0.45 | 6.9 | 0.00 | 0.45 | `r_640_4.c` |
| `E_16_64_4` | elementwise | 8 | 0.43 | 53.6 | 0.00 | 0.43 | `E_16_64_4.c` |
| `r_640_4_64_4` | reduce | 1 | 0.36 | 355.5 | 0.00 | 0.36 | `r_640_4_64_4.c` |
| `r_64_4_970` | reduce | 2 | 0.33 | 166.1 | 0.00 | 0.33 | `r_64_4_970.c` |
| `E_8_4` | elementwise | 48 | 0.30 | 6.3 | 0.00 | 0.30 | `E_8_4.c` |
| `rockchip_linear_1_2560_2560` | fused INT8 Linear | 1 | 0.24 | 235.7 | 0.22 | 0.01 | `rockchip_linear_1_2560_2560.c` |
| `E_8192_3` | elementwise | 24 | 0.19 | 8.1 | 0.00 | 0.19 | `E_8192_3.c` |
| `r_32_32_4` | reduce | 24 | 0.19 | 7.7 | 0.00 | 0.19 | `r_32_32_4.c` |
| `E_4_32_2_4_2` | elementwise | 8 | 0.07 | 8.9 | 0.00 | 0.07 | `E_4_32_2_4_2.c` |
| `r_16_64_4` | reduce | 8 | 0.07 | 8.5 | 0.00 | 0.07 | `r_16_64_4.c` |
| `E_16_32_2_4` | elementwise | 8 | 0.06 | 8.1 | 0.00 | 0.06 | `E_16_32_2_4.c` |
| `r_4_64_4` | reduce | 8 | 0.04 | 5.4 | 0.00 | 0.04 | `r_4_64_4.c` |
| `E` | elementwise | 9 | 0.04 | 4.6 | 0.00 | 0.04 | `E.c` |
| `r_64_4` | reduce | 2 | 0.01 | 5.8 | 0.00 | 0.01 | `r_64_4.c` |

## Class rollup

| class | ms | share |
| --- | --: | --: |
| fused INT8 Linear | 147.37 | 78.9% |
| elementwise | 20.04 | 10.7% |
| fill (kernel) | 9.43 | 5.0% |
| GDN scan | 5.22 | 2.8% |
| reduce | 2.58 | 1.4% |
| attention | 2.22 | 1.2% |

sum of kernels 186.86 ms; wall 196.18 ms; outside 9.32 ms
