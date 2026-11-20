# Decode kernels: source and in-model timing

One warm decode token, `MAX_CALLS_PER_SEG = 1`. Wall **187.58 ms**, 910 calls, 35 distinct kernels.

| kernel | class | calls | ms | us/call | submit ms | cpu ms | source |
| --- | --- | --: | --: | --: | --: | --: | --- |
| `rockchip_linear_1_9216_2560` | fused INT8 Linear | 64 | 49.78 | 777.8 | 48.64 | 1.14 | `rockchip_linear_1_9216_2560.c` |
| `rockchip_linear_1_2560_4096` | fused INT8 Linear | 96 | 40.01 | 416.8 | 38.58 | 1.43 | `rockchip_linear_1_2560_4096.c` |
| `rockchip_linear_1_24576_2560` | fused INT8 Linear | 10 | 20.19 | 2019.0 | 19.91 | 0.28 | `rockchip_linear_1_24576_2560.c` |
| `rockchip_linear_1_8192_2560` | fused INT8 Linear | 24 | 16.62 | 692.6 | 16.15 | 0.47 | `rockchip_linear_1_8192_2560.c` |
| `rockchip_linear_1_4096_2560` | fused INT8 Linear | 24 | 8.63 | 359.5 | 8.30 | 0.33 | `rockchip_linear_1_4096_2560.c` |
| `E_62080_4` | elementwise | 1 | 8.49 | 8487.7 | 0.00 | 8.49 | `E_62080_4.c` |
| `rockchip_linear_1_10240_2560` | fused INT8 Linear | 8 | 6.84 | 855.1 | 6.68 | 0.16 | `rockchip_linear_1_10240_2560.c` |
| `rockchip_scan_simd_1_32_1_128_128_1` | GDN scan | 24 | 5.61 | 233.7 | 0.00 | 5.61 | `rockchip_scan_simd_1_32_1_128_128_1.c` |
| `rockchip_linear_1_2560_1024` | fused INT8 Linear | 32 | 3.61 | 112.7 | 3.25 | 0.35 | `rockchip_linear_1_2560_1024.c` |
| `rockchip_attn_1_160` | attention | 8 | 2.32 | 290.0 | 0.00 | 2.32 | `rockchip_attn_1_160.c` |
| `E_124160_2` | elementwise | 1 | 2.10 | 2099.1 | 0.00 | 2.10 | `E_124160_2.c` |
| `E_2_16_32_4` | elementwise | 48 | 1.91 | 39.8 | 0.00 | 1.91 | `E_2_16_32_4.c` |
| `E_80_32_64_4` | elementwise | 1 | 1.60 | 1596.9 | 0.00 | 1.60 | `E_80_32_64_4.c` |
| `rockchip_linear_1_32_2560` | fused INT8 Linear | 48 | 1.44 | 29.9 | 0.92 | 0.51 | `rockchip_linear_1_32_2560.c` |
| `E_1024_4` | elementwise | 72 | 1.22 | 17.0 | 0.00 | 1.22 | `E_1024_4.c` |
| `E_3072_3` | elementwise | 32 | 1.18 | 37.0 | 0.00 | 1.18 | `E_3072_3.c` |
| `r_16_32_4` | reduce | 48 | 1.12 | 23.3 | 0.00 | 1.12 | `r_16_32_4.c` |
| `E_8_32_4_4` | elementwise | 24 | 1.07 | 44.7 | 0.00 | 1.07 | `E_8_32_4_4.c` |
| `E_81920_4` | fill (kernel) | 8 | 0.68 | 84.5 | 0.00 | 0.68 | `E_81920_4.c` |
| `E_8192_4` | elementwise | 24 | 0.65 | 27.1 | 0.00 | 0.65 | `E_8192_4.c` |
| `E_640_4` | elementwise | 97 | 0.64 | 6.6 | 0.00 | 0.64 | `E_640_4.c` |
| `r_640_4` | reduce | 65 | 0.46 | 7.1 | 0.00 | 0.46 | `r_640_4.c` |
| `E_16_64_4` | elementwise | 8 | 0.43 | 53.9 | 0.00 | 0.43 | `E_16_64_4.c` |
| `r_640_4_64_4` | reduce | 1 | 0.34 | 344.2 | 0.00 | 0.34 | `r_640_4_64_4.c` |
| `r_64_4_970` | reduce | 2 | 0.32 | 161.9 | 0.00 | 0.32 | `r_64_4_970.c` |
| `E_8_4` | elementwise | 48 | 0.30 | 6.2 | 0.00 | 0.30 | `E_8_4.c` |
| `rockchip_linear_1_2560_2560` | fused INT8 Linear | 1 | 0.24 | 240.3 | 0.23 | 0.01 | `rockchip_linear_1_2560_2560.c` |
| `E_8192_3` | elementwise | 24 | 0.20 | 8.2 | 0.00 | 0.20 | `E_8192_3.c` |
| `r_32_32_4` | reduce | 24 | 0.18 | 7.6 | 0.00 | 0.18 | `r_32_32_4.c` |
| `E_4_32_2_4_2` | elementwise | 8 | 0.07 | 9.1 | 0.00 | 0.07 | `E_4_32_2_4_2.c` |
| `r_16_64_4` | reduce | 8 | 0.07 | 8.7 | 0.00 | 0.07 | `r_16_64_4.c` |
| `E_16_32_2_4` | elementwise | 8 | 0.06 | 8.1 | 0.00 | 0.06 | `E_16_32_2_4.c` |
| `r_4_64_4` | reduce | 8 | 0.05 | 5.9 | 0.00 | 0.05 | `r_4_64_4.c` |
| `E` | elementwise | 9 | 0.04 | 4.9 | 0.00 | 0.04 | `E.c` |
| `r_64_4` | reduce | 2 | 0.01 | 5.7 | 0.00 | 0.01 | `r_64_4.c` |

## Class rollup

| class | ms | share |
| --- | --: | --: |
| fused INT8 Linear | 147.35 | 82.6% |
| elementwise | 19.98 | 11.2% |
| GDN scan | 5.61 | 3.1% |
| reduce | 2.56 | 1.4% |
| attention | 2.32 | 1.3% |
| fill (kernel) | 0.68 | 0.4% |

sum of kernels 178.50 ms; wall 187.58 ms; outside 9.09 ms
