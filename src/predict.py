"""
Rice Grain Quality Detection — Standalone Inference Script

Supports ONNX Runtime and TFLite inference engines.
The model runs entirely on-device with no cloud dependencies.

Usage:
    python src/predict.py --image path/to/grain.jpg --model models/edge/onnx/tinyricenet_int8.onnx
    python src/predict.py --image path/to/grain.jpg --model models/edge/tflite/tinyricenet_int8.tflite
    python src/predict.py --image path/to/grain.jpg --model models/edge/onnx/tinyricenet_int8.onnx --benchmark
"""

import argparse
import os
import time
import numpy as np
from pathlib import Path
from PIL import Image

CLASS_NAMES = ["broken_rice", "chalky_rice", "foreign_object", "head_rice", "unhulled_rice"]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

INPUT_SIZE = 128


def preprocess(image_path):
    """Load and preprocess a grain image for inference."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)  # HWC to CHW
    arr = np.expand_dims(arr, 0)  # Add batch dimension
    return arr


def softmax(logits):
    """Compute softmax probabilities from raw logits."""
    exp = np.exp(logits - np.max(logits))
    return exp / exp.sum()


def infer_onnx(model_path, input_array):
    """Run inference using ONNX Runtime."""
    import onnxruntime as ort

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    start = time.perf_counter()
    output = session.run(None, {input_name: input_array})
    elapsed_ms = (time.perf_counter() - start) * 1000

    return output[0][0], elapsed_ms


def infer_tflite(model_path, input_array):
    """Run inference using TFLite interpreter."""
    try:
        import tflite_runtime.interpreter as tflite
        interpreter = tflite.Interpreter(model_path=model_path)
    except ImportError:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=model_path)

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Handle quantized input
    input_dtype = input_details[0]["dtype"]
    if input_dtype == np.int8:
        scale, zero_point = input_details[0]["quantization"]
        input_array = (input_array / scale + zero_point).astype(np.int8)
    elif input_dtype == np.uint8:
        scale, zero_point = input_details[0]["quantization"]
        input_array = (input_array / scale + zero_point).astype(np.uint8)

    interpreter.set_tensor(input_details[0]["index"], input_array)

    start = time.perf_counter()
    interpreter.invoke()
    elapsed_ms = (time.perf_counter() - start) * 1000

    output = interpreter.get_tensor(output_details[0]["index"])

    # Dequantize output
    output_dtype = output_details[0]["dtype"]
    if output_dtype in (np.int8, np.uint8):
        scale, zero_point = output_details[0]["quantization"]
        output = (output.astype(np.float32) - zero_point) * scale

    return output[0], elapsed_ms


def run_benchmark(infer_fn, model_path, input_array, n_runs=200):
    """Run repeated inference to measure average latency."""
    # Warmup
    for _ in range(10):
        infer_fn(model_path, input_array)

    times = []
    for _ in range(n_runs):
        _, ms = infer_fn(model_path, input_array)
        times.append(ms)

    times = np.array(times)
    return {
        "mean_ms": float(np.mean(times)),
        "median_ms": float(np.median(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "p95_ms": float(np.percentile(times, 95)),
    }


def main():
    parser = argparse.ArgumentParser(description="Rice grain quality inference")
    parser.add_argument("--image", type=str, required=True, help="Path to grain image")
    parser.add_argument("--model", type=str, required=True, help="Path to model (.onnx or .tflite)")
    parser.add_argument("--benchmark", action="store_true", help="Run latency benchmark (200 iterations)")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    model_ext = Path(args.model).suffix.lower()
    model_size_kb = os.path.getsize(args.model) / 1024

    if model_ext == ".onnx":
        engine = "ONNX Runtime"
        infer_fn = infer_onnx
    elif model_ext == ".tflite":
        engine = "TFLite"
        infer_fn = infer_tflite
    else:
        raise ValueError(f"Unsupported model format: {model_ext}")

    input_array = preprocess(args.image)
    logits, inference_ms = infer_fn(args.model, input_array)
    probs = softmax(logits)

    top_idx = np.argmax(probs)
    predicted_class = CLASS_NAMES[top_idx]
    confidence = probs[top_idx]

    print(f"\nRice Grain Quality Detection")
    print(f"  Engine:      {engine}")
    print(f"  Model:       {args.model}")
    print(f"  Model size:  {model_size_kb:.1f} KB")
    print(f"  Image:       {args.image}")
    print(f"  Inference:   {inference_ms:.2f} ms")
    print(f"")
    print(f"  Prediction:  {predicted_class}")
    print(f"  Confidence:  {confidence * 100:.1f}%")
    print(f"")
    print(f"  All classes:")

    ranked = np.argsort(probs)[::-1]
    for idx in ranked:
        bar_len = int(probs[idx] * 30)
        bar = "#" * bar_len
        print(f"    {CLASS_NAMES[idx]:<20s} {probs[idx] * 100:5.1f}%  {bar}")

    if args.benchmark:
        print(f"\n  Latency benchmark ({engine}, 200 runs):")
        stats = run_benchmark(infer_fn, args.model, input_array)
        print(f"    Mean:   {stats['mean_ms']:.2f} ms")
        print(f"    Median: {stats['median_ms']:.2f} ms")
        print(f"    Std:    {stats['std_ms']:.2f} ms")
        print(f"    P95:    {stats['p95_ms']:.2f} ms")
        print(f"    Min:    {stats['min_ms']:.2f} ms")
        print(f"    Max:    {stats['max_ms']:.2f} ms")


if __name__ == "__main__":
    main()