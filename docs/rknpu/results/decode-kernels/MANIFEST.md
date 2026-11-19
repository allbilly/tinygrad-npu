# Decode kernels: source and in-model timing

One warm decode token, `MAX_CALLS_PER_SEG = 1`. Wall **190.39 ms**, 956 calls, 35 distinct kernels.

| kernel | class | calls | ms | us/call | submit ms | cpu ms | source |
| --- | --- | --: | --: | --: | --: | --: | --- |
| `rockchip_linear_1_9216_2560` | fused INT8 Linear | 64 | 49.81 | 778.3 | 48.65 | 1.16 | `rockchip_linear_1_9216_2560.c` |
| `rockchip_linear_1_2560_4096` | fused INT8 Linear | 96 | 40.03 | 417.0 | 38.61 | 1.42 | `rockchip_linear_1_2560_4096.c` |
| `rockchip_linear_1_24576_2560` | fused INT8 Linear | 10 | 20.25 | 2025.2 | 19.97 | 0.29 | `rockchip_linear_1_24576_2560.c` |
| `rockchip_linear_1_8192_2560` | fused INT8 Linear | 24 | 16.68 | 694.9 | 16.20 | 0.48 | `rockchip_linear_1_8192_2560.c` |
| `E_131072_4` | fill (kernel) | 48 | 8.94 | 186.2 | 0.00 | 8.94 | `E_131072_4.c` |
| `rockchip_linear_1_4096_2560` | fused INT8 Linear | 24 | 8.61 | 358.7 | 8.29 | 0.32 | `rockchip_linear_1_4096_2560.c` |
| `rockchip_linear_1_10240_2560` | fused INT8 Linear | 8 | 6.83 | 853.3 | 6.66 | 0.17 | `rockchip_linear_1_10240_2560.c` |
| `rockchip_scan_simd_1_32_1_128_128_1` | GDN scan | 24 | 5.37 | 223.6 | 0.00 | 5.37 | `rockchip_scan_simd_1_32_1_128_128_1.c` |
| `rockchip_linear_1_2560_1024` | fused INT8 Linear | 32 | 3.62 | 113.2 | 3.26 | 0.36 | `rockchip_linear_1_2560_1024.c` |
| `E_2_16_32_4` | elementwise | 48 | 3.47 | 72.3 | 0.00 | 3.47 | `E_2_16_32_4.c` |
| `rockchip_attn_1_160` | attention | 8 | 2.37 | 296.0 | 0.00 | 2.37 | `rockchip_attn_1_160.c` |
| `E_1024_4` | elementwise | 72 | 1.85 | 25.7 | 0.00 | 1.85 | `E_1024_4.c` |
| `E_80_32_64_4` | elementwise | 1 | 1.59 | 1588.4 | 0.00 | 1.59 | `E_80_32_64_4.c` |
| `rockchip_linear_1_32_2560` | fused INT8 Linear | 48 | 1.43 | 29.7 | 0.93 | 0.50 | `rockchip_linear_1_32_2560.c` |
| `r_16_32_4` | reduce | 48 | 1.27 | 26.5 | 0.00 | 1.27 | `r_16_32_4.c` |
| `E_3072_3` | elementwise | 32 | 1.18 | 36.8 | 0.00 | 1.18 | `E_3072_3.c` |
| `E_8_32_4_4` | elementwise | 24 | 1.10 | 46.0 | 0.00 | 1.10 | `E_8_32_4_4.c` |
| `E_640_4` | elementwise | 97 | 1.07 | 11.0 | 0.00 | 1.07 | `E_640_4.c` |
| `E_8192_4` | elementwise | 24 | 0.74 | 30.6 | 0.00 | 0.74 | `E_8192_4.c` |
| `E_81920_4` | fill (kernel) | 8 | 0.71 | 88.8 | 0.00 | 0.71 | `E_81920_4.c` |
| `r_640_4` | reduce | 65 | 0.45 | 6.9 | 0.00 | 0.45 | `r_640_4.c` |
| `E_16_64_4` | elementwise | 8 | 0.43 | 53.2 | 0.00 | 0.43 | `E_16_64_4.c` |
| `E_62080_4` | elementwise | 1 | 0.41 | 411.0 | 0.00 | 0.41 | `E_62080_4.c` |
| `r_640_4_64_4` | reduce | 1 | 0.36 | 355.8 | 0.00 | 0.36 | `r_640_4_64_4.c` |
| `r_64_4_970` | reduce | 2 | 0.33 | 167.0 | 0.00 | 0.33 | `r_64_4_970.c` |
| `E_8_4` | elementwise | 48 | 0.29 | 6.1 | 0.00 | 0.29 | `E_8_4.c` |
| `rockchip_linear_1_2560_2560` | fused INT8 Linear | 1 | 0.24 | 236.8 | 0.22 | 0.01 | `rockchip_linear_1_2560_2560.c` |
| `E_8192_3` | elementwise | 24 | 0.19 | 8.0 | 0.00 | 0.19 | `E_8192_3.c` |
| `r_32_32_4` | reduce | 24 | 0.18 | 7.7 | 0.00 | 0.18 | `r_32_32_4.c` |
| `E_16_32_2_4` | elementwise | 8 | 0.18 | 22.0 | 0.00 | 0.18 | `E_16_32_2_4.c` |
| `E_4_32_2_4_2` | elementwise | 8 | 0.08 | 9.7 | 0.00 | 0.08 | `E_4_32_2_4_2.c` |
| `r_16_64_4` | reduce | 8 | 0.07 | 8.5 | 0.00 | 0.07 | `r_16_64_4.c` |
| `r_4_64_4` | reduce | 8 | 0.04 | 5.4 | 0.00 | 0.04 | `r_4_64_4.c` |
| `E` | elementwise | 8 | 0.04 | 4.5 | 0.00 | 0.04 | `E.c` |
| `r_64_4` | reduce | 2 | 0.01 | 6.4 | 0.00 | 0.01 | `r_64_4.c` |

## Class rollup

| class | ms | share |
| --- | --: | --: |
| fused INT8 Linear | 147.50 | 81.9% |
| elementwise | 12.61 | 7.0% |
| fill (kernel) | 9.65 | 5.4% |
| GDN scan | 5.37 | 3.0% |
| reduce | 2.72 | 1.5% |
| attention | 2.37 | 1.3% |

sum of kernels 180.20 ms; wall 190.39 ms; outside 10.19 ms
