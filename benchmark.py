import asyncio
import time
from typing import Any, Dict, List

import aiohttp
import numpy as np
import psutil
import json

class BenchmarkRunner:
    def __init__(self, url: str):
        self.base_url = url
        self.session = aiohttp.ClientSession()

    async def get_embeddings(self, texts: List[str]) -> Dict[str, Any]:
        url = f"{self.base_url}/embed"

        data = {"texts": texts}

        start_time = time.time()

        try:
            async with self.session.post(url, json=data) as response:
                response_time = time.time() - start_time

                if response.status == 200:
                    result = await response.json()
                    return {
                        "success": True,
                        "response_time": response_time,
                        "embeddings_count": len(result["embeddings"]),
                    }
                else:
                    return {
                        "success": False,
                        "response_time": response_time,
                        "error": f"HTTP {response.status}",
                    }
        except Exception as e:
            return {
                "success": False,
                "response_time": time.time() - start_time,
                "error": str(e),
            }

    async def warmup(
        self, texts_pool: List[str], texts_per_request: int, iterations: int = 5
    ):
        tasks = []
        for _ in range(iterations):
            tasks.append(self.get_embeddings(texts_pool))

        await asyncio.gather(*tasks)

    async def run_benchmark(
        self,
        texts_pool: List[str],
        texts_per_request: int = 1,
        num_requests: int = 100,
        concurrency: int = 10,
        request_delay: float = 0.0,
    ) -> Dict[str, Any]:
        async def worker(worker_id: int):
            results = []
            requests_per_worker = num_requests // concurrency

            for i in range(requests_per_worker):
                start_idx = (
                    (worker_id * requests_per_worker + i)
                    * texts_per_request
                    % len(texts_pool)
                )
                indices = [
                    (start_idx + j) % len(texts_pool) for j in range(texts_per_request)
                ]
                selected_texts = [texts_pool[idx] for idx in indices]

                result = await self.get_embeddings(selected_texts)

                cpu_after = psutil.cpu_percent(interval=None)
                mem_after = psutil.virtual_memory().percent

                result.update(
                    {
                        "worker_id": worker_id,
                        "request_num": i,
                        "cpu_after": cpu_after,
                        "mem_after": mem_after,
                    }
                )

                results.append(result)

                if request_delay > 0:
                    await asyncio.sleep(request_delay)

            return results

        start_time = time.time()
        tasks = [worker(i) for i in range(concurrency)]
        worker_results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time

        all_results = []
        for wr in worker_results:
            all_results.extend(wr)

        successful = [r for r in all_results if r["success"]]
        failed = [r for r in all_results if not r["success"]]

        if successful:
            response_times = [r["response_time"] * 1000 for r in successful]

            latency_metrics = {
                "min_ms": float(min(response_times)),
                "max_ms": float(max(response_times)),
                "mean_ms": float(np.mean(response_times)),
                "median_ms": float(np.median(response_times)),
                "p95_ms": float(np.percentile(response_times, 95)),
                "p99_ms": float(np.percentile(response_times, 99)),
                "std_ms": float(np.std(response_times)),
            }

            throughput = len(successful) / total_time  # rps
        else:
            latency_metrics = {}
            throughput = 0

        cpu_usage = []
        mem_usage = []
        for r in successful:
            if "cpu_after" in r:
                cpu_usage.append(r["cpu_after"])
            if "mem_after" in r:
                mem_usage.append(r["mem_after"])

        resource_metrics = {}
        if cpu_usage:
            resource_metrics.update(
                {
                    "cpu_mean_percent": float(np.mean(cpu_usage)),
                    "cpu_max_percent": float(max(cpu_usage)),
                    "cpu_min_percent": float(min(cpu_usage)),
                    "cpu_std_percent": float(np.std(cpu_usage)),
                }
            )

        if mem_usage:
            resource_metrics.update(
                {
                    "memory_mean_percent": float(np.mean(mem_usage)),
                    "memory_max_percent": float(max(mem_usage)),
                    "memory_min_percent": float(min(mem_usage)),
                    "memory_std_percent": float(np.std(mem_usage)),
                }
            )

        return {
            "config": {
                "num_requests": num_requests,
                "concurrency": concurrency,
                "texts_per_request": texts_per_request,
                "request_delay": request_delay,
            },
            "summary": {
                "total_time_sec": total_time,
                "successful_requests": len(successful),
                "failed_requests": len(failed),
                "success_rate": (
                    len(successful) / num_requests * 100 if num_requests > 0 else 0
                ),
                "throughput_rps": throughput,
            },
            "latency": latency_metrics,
            "resources": resource_metrics,
        }


TEXTS_POOL = [
    "Цель этого ДЗ",
    "Изучить разные уровни оптимизации inference pipeline и проанализировать эффекты. ",
    "Сетап",
    "Для работы используйте модель для расчетов эмбеддингов – rubert-mini-frida. ",
    "Домашнее задание будет состоять из трёх частей:",
    "базовый инференс (2 балла);",
    "rонвертация модели в ONNX –  оптимизация модели (3 балла); ",
    "динамическое батчирование – оптимизация сервинга (5 баллов). ",
    "Что вам нужно сделать",
    "Часть 1 — Базовый инференс ",
    "1) Загрузите модель rubert-mini-frida с hf.",
    "2) Реализуйте простой инференс, используя transformers/sentence_transformers (поднимаем все на CPU).",
    "3) Упакуйте в HTTP сервис (например, через FastAPI).",
    "4) Замерьте бенчмарки. Метрики вы выбираете самостоятельно, выбор метрик нужно обосновать в отчёте. Смотрите в сторону latency, throughput, потребление ресурсов.",
    "5) Зафиксируйте значения в отчете.",
    "Часть 2 — Конвертация модели в ONNX (оптимизация модели) ",
    "1) Сконвертируйте модель в onnx.",
    "2) Реализуйте инференс через onnx-runtime.",
    "3) Упакуйте в HTTP сервис (например, через FastAPI).",
    "4) Замерьте бенчмарки. Метрики вы выбираете самостоятельно, выбор метрик нужно обосновать. Смотрите в сторону latency, throughput, потребление ресурсов.",
    "5) Проанализируйте, сравните с частью 1, зафиксируйте анализ в отчете.",
    "Часть 3 — Динамическое батчирование (оптимизация сервинга)",
    "Идея: собираем несколько запросов за небольшой промежуток времени и объединяем их в один batch перед инференсом.",
    "1) Реализуйте динамическое батчирование для инференса модели из части 2. Реализацию можно сделать через очередь + worker.",
    "2) Замерьте бенчмарки. Метрики вы выбираете самостоятельно, выбор метрик нужно обосновать. ",
    "3) Проанализируйте, сравните с инференсом без динамического батчирования из части 2, зафиксируйте в отчете.",
    "Что и как вы отправляете на проверку",
    "Ссылка на работающий код, который можно запустить и проверить, и отчет.",
]


async def main():
    benchmark = BenchmarkRunner("http://localhost:8000")

    await benchmark.warmup(TEXTS_POOL[:5], 1, iterations=5)

    results = await benchmark.run_benchmark(
        texts_pool=TEXTS_POOL,
        texts_per_request=1,
        num_requests=1000,
        concurrency=5,
        request_delay=0,
    )

    print(json.dumps(results, indent=4))

    await benchmark.session.close()


if __name__ == "__main__":
    asyncio.run(main())
