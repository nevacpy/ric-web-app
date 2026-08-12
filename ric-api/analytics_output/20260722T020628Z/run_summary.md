# Analytics Summary: 20260722T020628Z

Source CSV: `/Users/NC1/Project/fyp-web-app/ric-api/compare_output/compare_results_20260722T020628Z.csv`

- Total rows: 2992
- Unique images: 187
- Overall accuracy: 0.101

## Best model+preprocess combinations

| combo | accuracy | median_latency_ms | p95_latency_ms | n |
|---|---:|---:|---:|---:|
| ric5-v1 | clahe_canny | 0.230 | 42.59 | 47.21 | 187 |
| ric1-v5 | clahe_canny | 0.214 | 42.60 | 48.07 | 187 |
| ric1-v5 | clahe | 0.209 | 42.66 | 46.54 | 187 |
| ric1-v5 | baseline | 0.118 | 42.53 | 48.52 | 187 |
| ric5-v1 | clahe | 0.118 | 42.60 | 46.90 | 187 |
| ric5-v1 | canny | 0.112 | 42.49 | 47.15 | 187 |
| ric1-v5 | canny | 0.091 | 42.61 | 46.38 | 187 |
| ric1-ric5-v6 | canny | 0.086 | 42.67 | 46.67 | 187 |
| ric1-ric5-v6 | clahe_canny | 0.080 | 42.47 | 46.84 | 187 |
| ric1-ric5-v6 | baseline | 0.080 | 42.50 | 46.53 | 187 |

## Hardest variants

| variant | accuracy | n |
|---|---:|---:|
| pet1_not_transparnet_embossed | 0.013 | 80 |
| pp5_not_transluent_embossed | 0.056 | 288 |
| pet1_print | 0.056 | 304 |
| pp5_printed | 0.062 | 128 |
| pet1_special | 0.062 | 64 |
| pet1_transparent_embossed | 0.063 | 1104 |
| pp5_special | 0.080 | 176 |
| pp5_transluent_embossed | 0.090 | 720 |
| pet1 | 0.438 | 16 |
| other_rics | 0.902 | 112 |
