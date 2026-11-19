# Decode kernels: source and in-model timing

One warm decode token, `MAX_CALLS_PER_SEG = 1`. Wall **199.04 ms**, 958 calls, 36 distinct kernels.

| kernel | class | calls | ms | us/call | submit ms | cpu ms | source |
| --- | --- | --: | --: | --: | --: | --: | --- |
| `rockchip_linear_1_9216_2560` | fused INT8 Linear | 64 | 49.90 | 779.7 | 48.74 | 1.15 | `rockchip_linear_1_9216_2560.c` |
| `rockchip_linear_1_2560_4096` | fused INT8 Linear | 96 | 40.04 | 417.1 | 38.58 | 1.46 | `rockchip_linear_1_2560_4096.c` |
| `rockchip_linear_1_24576_2560` | fused INT8 Linear | 10 | 20.20 | 2020.1 | 19.92 | 0.29 | `rockchip_linear_1_24576_2560.c` |
| `rockchip_linear_1_8192_2560` | fused INT8 Linear | 24 | 16.65 | 693.7 | 16.19 | 0.46 | `rockchip_linear_1_8192_2560.c` |
| `E_131072_4` | fill (kernel) | 48 | 8.83 | 184.0 | 0.00 | 8.83 | `E_131072_4.c` |
| `rockchip_linear_1_4096_2560` | fused INT8 Linear | 24 | 8.62 | 359.2 | 8.30 | 0.32 | `rockchip_linear_1_4096_2560.c` |
| `E_62080_4` | elementwise | 1 | 8.53 | 8525.9 | 0.00 | 8.53 | `E_62080_4.c` |
| `rockchip_linear_1_10240_2560` | fused INT8 Linear | 8 | 6.86 | 858.0 | 6.71 | 0.15 | `rockchip_linear_1_10240_2560.c` |
| `rockchip_scan_simd_1_32_1_128_128_1` | GDN scan | 24 | 5.23 | 217.9 | 0.00 | 5.23 | `rockchip_scan_simd_1_32_1_128_128_1.c` |
| `rockchip_linear_1_2560_1024` | fused INT8 Linear | 32 | 3.63 | 113.6 | 3.27 | 0.36 | `rockchip_linear_1_2560_1024.c` |
| `E_2_16_32_4` | elementwise | 48 | 3.47 | 72.4 | 0.00 | 3.47 | `E_2_16_32_4.c` |
| `rockchip_attn_1_160` | attention | 8 | 2.21 | 276.2 | 0.00 | 2.21 | `rockchip_attn_1_160.c` |
| `E_124160_2` | elementwise | 1 | 2.10 | 2104.9 | 0.00 | 2.10 | `E_124160_2.c` |
| `E_1024_4` | elementwise | 72 | 1.87 | 25.9 | 0.00 | 1.87 | `E_1024_4.c` |
| `E_80_32_64_4` | elementwise | 1 | 1.60 | 1596.3 | 0.00 | 1.60 | `E_80_32_64_4.c` |
| `rockchip_linear_1_32_2560` | fused INT8 Linear | 48 | 1.44 | 30.1 | 0.92 | 0.52 | `rockchip_linear_1_32_2560.c` |
| `r_16_32_4` | reduce | 48 | 1.26 | 26.3 | 0.00 | 1.26 | `r_16_32_4.c` |
| `E_3072_3` | elementwise | 32 | 1.18 | 36.9 | 0.00 | 1.18 | `E_3072_3.c` |
| `E_8_32_4_4` | elementwise | 24 | 1.11 | 46.3 | 0.00 | 1.11 | `E_8_32_4_4.c` |
| `E_640_4` | elementwise | 97 | 1.07 | 11.0 | 0.00 | 1.07 | `E_640_4.c` |
| `E_81920_4` | fill (kernel) | 8 | 0.69 | 85.8 | 0.00 | 0.69 | `E_81920_4.c` |
| `E_8192_4` | elementwise | 24 | 0.66 | 27.5 | 0.00 | 0.66 | `E_8192_4.c` |
| `r_640_4` | reduce | 65 | 0.45 | 7.0 | 0.00 | 0.45 | `r_640_4.c` |
| `E_16_64_4` | elementwise | 8 | 0.42 | 52.9 | 0.00 | 0.42 | `E_16_64_4.c` |
| `r_640_4_64_4` | reduce | 1 | 0.35 | 354.7 | 0.00 | 0.35 | `r_640_4_64_4.c` |
| `r_64_4_970` | reduce | 2 | 0.31 | 153.7 | 0.00 | 0.31 | `r_64_4_970.c` |
| `E_8_4` | elementwise | 48 | 0.30 | 6.3 | 0.00 | 0.30 | `E_8_4.c` |
| `rockchip_linear_1_2560_2560` | fused INT8 Linear | 1 | 0.24 | 242.1 | 0.23 | 0.01 | `rockchip_linear_1_2560_2560.c` |
| `E_8192_3` | elementwise | 24 | 0.19 | 8.1 | 0.00 | 0.19 | `E_8192_3.c` |
| `r_32_32_4` | reduce | 24 | 0.19 | 7.7 | 0.00 | 0.19 | `r_32_32_4.c` |
| `E_16_32_2_4` | elementwise | 8 | 0.18 | 22.1 | 0.00 | 0.18 | `E_16_32_2_4.c` |
| `E_4_32_2_4_2` | elementwise | 8 | 0.08 | 9.7 | 0.00 | 0.08 | `E_4_32_2_4_2.c` |
| `r_16_64_4` | reduce | 8 | 0.07 | 8.4 | 0.00 | 0.07 | `r_16_64_4.c` |
| `r_4_64_4` | reduce | 8 | 0.05 | 5.7 | 0.00 | 0.05 | `r_4_64_4.c` |
| `E` | elementwise | 9 | 0.04 | 4.6 | 0.00 | 0.04 | `E.c` |
| `r_64_4` | reduce | 2 | 0.01 | 6.6 | 0.00 | 0.01 | `r_64_4.c` |

## Class rollup

| class | ms | share |
| --- | --: | --: |
| fused INT8 Linear | 147.59 | 77.7% |
| elementwise | 22.81 | 12.0% |
| fill (kernel) | 9.52 | 5.0% |
| GDN scan | 5.23 | 2.8% |
| reduce | 2.69 | 1.4% |
| attention | 2.21 | 1.2% |

sum of kernels 190.04 ms; wall 199.04 ms; outside 8.99 ms
