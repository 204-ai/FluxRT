# FluxRT Benchmark Report

## Configuration

```json
{
  "default_prompt": "Turn this into art.",
  "default_steps": 2,
  "default_seed": 52,
  "models_path": "FLUX.2-klein-4B",
  "int8_models_path": "FLUX.2-klein-4B-int8",
  "resolution": {
    "height": 320,
    "width": 576
  },
  "compile_models": true,
  "enable_spatial_cache": true,
  "enable_int8_quantization": false,
  "enable_tiny_vae": true,
  "enable_flow_upscaler": true,
  "target_fps": null,
  "interpolation_exp": 2,
  "use_reference_image": false,
  "logging": false
}
```

## Hardware Information

```json
{
  "platform": "Linux-6.19.6-arch1-1-x86_64-with-glibc2.43",
  "python": "3.12.13",
  "cpu": "AMD Ryzen 9 9950X 16-Core Processor",
  "cpu_cores_logical": 32,
  "gpu": [
    {
      "name": "NVIDIA GeForce RTX 5090",
      "vram_gb": 31.36,
      "cc": "12.0"
    }
  ]
}
```

## Results

| Dynamic Area | Processing Time (s) | FPS |
|-------------:|--------------------:|----:|
| 0% | 0.0854 | 46.98 |
| 10% | 0.1059 | 38.52 |
| 25% | 0.1285 | 31.19 |
| 50% | 0.1551 | 25.92 |
| 75% | 0.1840 | 22.03 |
| 90% | 0.1884 | 21.23 |
| 100% | 0.1949 | 20.52 |

**End-to-end latency:** 0.1911 s

**Reserved GPU memory:** 22.3086 GB