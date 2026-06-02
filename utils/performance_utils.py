import argparse
import json
import os
import time

_MIB = 1024 ** 2


def reset_cuda_peak_memory():
    import torch

    for device_index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(device_index)


def synchronize_cuda():
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def cuda_peak_memory():
    import torch

    devices = {}
    for device_index in range(torch.cuda.device_count()):
        devices[f"cuda:{device_index}"] = {
            "name": torch.cuda.get_device_name(device_index),
            "max_memory_allocated_mib": torch.cuda.max_memory_allocated(device_index) / _MIB,
            "max_memory_reserved_mib": torch.cuda.max_memory_reserved(device_index) / _MIB,
        }
    return devices


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as metrics_file:
        json.dump(payload, metrics_file, indent=2)


def save_training_metrics(model_path, stage, start_time, iterations):
    synchronize_cuda()
    elapsed_seconds = time.perf_counter() - start_time
    metrics = {
        "stage": stage,
        "iterations": iterations,
        "training_seconds": elapsed_seconds,
        "training_minutes": elapsed_seconds / 60.0,
        "cuda_peak_memory": cuda_peak_memory(),
    }
    output_path = os.path.join(model_path, f"training_metrics_{stage}.json")
    write_json(output_path, metrics)
    print(f"Saved {stage} performance metrics to {output_path}")
    return metrics


def aggregate_training_metrics(stage1_path, stage2_path, output_path):
    with open(stage1_path, "r", encoding="utf-8") as stage1_file:
        stage1 = json.load(stage1_file)
    with open(stage2_path, "r", encoding="utf-8") as stage2_file:
        stage2 = json.load(stage2_file)

    total_seconds = float(stage1["training_seconds"]) + float(stage2["training_seconds"])
    devices = {}
    for stage_metrics in (stage1, stage2):
        for device, memory in stage_metrics.get("cuda_peak_memory", {}).items():
            current = devices.setdefault(device, {"name": memory.get("name", "")})
            for metric_name in ("max_memory_allocated_mib", "max_memory_reserved_mib"):
                current[metric_name] = max(
                    float(current.get(metric_name, 0.0)),
                    float(memory.get(metric_name, 0.0)),
                )

    metrics = {
        "training_seconds": total_seconds,
        "training_minutes": total_seconds / 60.0,
        "stages": {"stage1": stage1, "stage2": stage2},
        "cuda_peak_memory_across_training": devices,
    }
    write_json(output_path, metrics)
    print(f"Saved total training performance metrics to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    aggregate_parser = subparsers.add_parser("aggregate-training")
    aggregate_parser.add_argument("stage1_path")
    aggregate_parser.add_argument("stage2_path")
    aggregate_parser.add_argument("output_path")
    args = parser.parse_args()

    if args.command == "aggregate-training":
        aggregate_training_metrics(args.stage1_path, args.stage2_path, args.output_path)


if __name__ == "__main__":
    main()
