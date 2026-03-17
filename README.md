# inference_optimizations

## Базовый инференс

```json
{
    "config": {
        "num_requests": 1000,
        "concurrency": 5,
        "texts_per_request": 1,
        "request_delay": 0
    },
    "summary": {
        "total_time_sec": 3.891538143157959,
        "successful_requests": 1000,
        "failed_requests": 0,
        "success_rate": 100.0,
        "throughput_rps": 256.96780121715733
    },
    "latency": {
        "min_ms": 3.9870738983154297,
        "max_ms": 46.048879623413086,
        "mean_ms": 19.273369073867798,
        "median_ms": 18.45395565032959,
        "p95_ms": 24.936151504516587,
        "p99_ms": 37.842345237731934,
        "std_ms": 3.3982217496045717
    },
    "resources": {
        "cpu_mean_percent": 0.38930000000000003,
        "cpu_max_percent": 50.0,
        "cpu_min_percent": 0.0,
        "cpu_std_percent": 3.3914149716600592,
        "memory_mean_percent": 53.99349999999999,
        "memory_max_percent": 54.3,
        "memory_min_percent": 53.7,
        "memory_std_percent": 0.18489929691591567
    }
}
```

