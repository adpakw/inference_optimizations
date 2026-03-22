# inference_optimizations

## Описание 
Реализованы три варианта сервиса эмбеддингов:
- Baseline - базовый инференс с использованием sentence-transformers
- ONNX - инференс с ONNX Runtime после конвертации модели
- Batch - динамическое батчирование запросов поверх ONNX Runtime

## Конвертация модели в ONNX
Перед запуском ONNX и Batch сервисов необходимо сконвертировать модель:
```bash
make convert_onnx
```

## Запуск бенчмарка
Для запуска бенчмарка необходимо сначала поднять сервис, затем запустить бенчмарк в другом терминале
```bash
# Базовый инференс (sentence-transformers)
make run_baseline

# ONNX Runtime инференс
make run_onnx

# Динамическое батчирование
make run_batch

# Запуск бенчмарка в другом терминале
make benchmark
```

## Обоснование выбора метрик
**Latency (задержка)**
- **min/mean/median/p95/p99** - показывают распределение времени ответа. p95 и p99 критичны для оценки качества сервиса под нагрузкой
- **std** - стандартное отклонение, показывает стабильность времени ответа

**Throughput (пропускная способность)**
- **throughput_rps (requests per second)** - количество успешных запросов в секунду, ключевая метрика производительности

**Resource Usage (потребление ресурсов)**
- **CPU (mean/max/std)** - нагрузка на процессор, пиковые значения могут указывать на узкие места системы, в которых возможны bottleneck'и
- **Memory (mean/max)** - потребление RAM, пиковые значения могут указывать на узкие места системы, в которых возможны bottleneck'и

## Результаты бенчмарков
**Условия тестирования:**
- Количество запросов: 1000
- Concurrency: 5
- Текстов на запрос: 1
- Задержка между запросами: 0

### Базовый инференс
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

### ONNX
```json
{
    "config": {
        "num_requests": 1000,
        "concurrency": 5,
        "texts_per_request": 1,
        "request_delay": 0
    },
    "summary": {
        "total_time_sec": 1.5313029289245605,
        "successful_requests": 1000,
        "failed_requests": 0,
        "success_rate": 100.0,
        "throughput_rps": 653.0386516678992
    },
    "latency": {
        "min_ms": 1.3790130615234375,
        "max_ms": 20.68614959716797,
        "mean_ms": 7.501349925994873,
        "median_ms": 7.386445999145508,
        "p95_ms": 8.8165283203125,
        "p99_ms": 9.77056503295898,
        "std_ms": 1.1554511520920467
    },
    "resources": {
        "cpu_mean_percent": 0.3353,
        "cpu_max_percent": 100.0,
        "cpu_min_percent": 0.0,
        "cpu_std_percent": 4.621738191416732,
        "memory_mean_percent": 48.1675,
        "memory_max_percent": 48.3,
        "memory_min_percent": 48.1,
        "memory_std_percent": 0.09451851670439829
    }
}
```

### Dynamic Batching
```json
{
    "config": {
        "num_requests": 1000,
        "concurrency": 5,
        "texts_per_request": 1,
        "request_delay": 0
    },
    "summary": {
        "total_time_sec": 104.98275589942932,
        "successful_requests": 1000,
        "failed_requests": 0,
        "success_rate": 100.0,
        "throughput_rps": 9.525373871476315
    },
    "latency": {
        "min_ms": 509.92393493652344,
        "max_ms": 548.0098724365234,
        "mean_ms": 524.5725107192993,
        "median_ms": 524.6530771255493,
        "p95_ms": 530.4845333099365,
        "p99_ms": 531.3290119171143,
        "std_ms": 4.362143355145866
    },
    "resources": {
        "cpu_mean_percent": 7.609499999999999,
        "cpu_max_percent": 100.0,
        "cpu_min_percent": 0.0,
        "cpu_std_percent": 23.325413174261243,
        "memory_mean_percent": 50.5511,
        "memory_max_percent": 52.5,
        "memory_min_percent": 49.8,
        "memory_std_percent": 0.5240789921376355
    }
}
```

## Вывод по результатам
**Базовый инференс**
- Наименьшая пропускная способность(throughput) среди всех реализаций
- Высокая задержка (mean ~19ms) обусловлена накладными расходами PyTorch и Python
- Нестабильность в p99 (37.8ms) указывает на возможные проблемы с GC

**ONNX**
- Throughput увеличился в 2.5x по сравнению с *baseline*
- Latency снизилась в 2.6x (с 19.3ms до 7.5ms)
- Значительно более стабильные p95/p99 (9.8ms vs 37.8ms)
- Снижение потребления памяти (~5%)
- Пиковые CPU нагрузки достигают 100% - ONNX эффективно утилизирует ресурсы

**Dynamic Batching**
- Снижение производительности - throughput упал в ~68x
- Latency выросла до ~525ms
- Высокая стабильность задержек (std ~4.4ms) - все запросы обрабатываются равномерно
- Все обусловлено тем, что мы собираем батч либо по времени, либо по размеру

**Вывод**
- Baseline не рекомендуется для production из-за низкой эффективности
- Для production больше подходт вариант с ONNX инференсом
- Для систем с пакетной обработкой подходит Dynamic Batching с оптимизированными параметрами