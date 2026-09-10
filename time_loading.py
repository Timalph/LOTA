"""
Standalone timing script for the training data pipeline in loader.py.

Loads `x` images the same way GenerativeImageTrainingSet does, but breaks
each sample into its individual stages and times them separately:

    1. disk read      -> open(path, 'rb') + PIL.Image.open().convert('RGB')
                          (+ JPEG re-compression when --unbiased is set)
    2. bit_patch step  -> bit_patch(...) (only when --isPatch, the default)
                          or plain Resize otherwise
    3. ToTensor
    4. Normalize

It reuses the real dataset/option machinery from config.py and loader.py so
the timings reflect exactly what happens during training.

Usage:
    python time_loading.py --dataset_name Wukong --num_images 100
    python time_loading.py --num_images 50 --isPatch False   # skip bit_patch
"""
import argparse
import statistics
from time import perf_counter

import numpy as np
from torchvision import transforms

from config import ConfigurationManager
from loader import (
    MODEL_NAME_MAP,
    GenerativeImageTrainingSet,
    bit_patch_process,
)


def build_options():
    parser = argparse.ArgumentParser(
        description='Time the loader.py preprocessing pipeline per image',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    cm = ConfigurationManager()
    parser = cm.define_arguments(parser)

    parser.add_argument('--num_images', type=int, default=50,
                         help='How many images to time')
    parser.add_argument('--dataset_name', type=str, default=None,
                         choices=list(MODEL_NAME_MAP.values()),
                         help='Which dataset folder to sample from '
                              '(defaults to the first one selected in --choices)')

    options = parser.parse_args()

    if options.unbiased:
        options.qf = 96

    if options.dataset_name is None:
        for idx, flag in enumerate(options.choices):
            if flag:
                options.dataset_name = MODEL_NAME_MAP[idx]
                break
        if options.dataset_name is None:
            raise ValueError("No dataset selected: pass --dataset_name or set --choices")

    return options


def time_single_image(dataset, index, options):
    """Times each stage of loading+preprocessing a single sample. Returns a dict of timings (seconds)."""
    img_path = dataset.all_images[index]
    label = dataset.labels[index]

    t0 = perf_counter()
    img = dataset._load_rgb(img_path, label)
    t1 = perf_counter()

    if options.isPatch:
        processed = bit_patch_process(
            img, options.img_height, options.bit_mode,
            options.patch_size, options.patch_mode
        )
        # bit_patch returns a numpy array; ToTensor needs that or a PIL image, both work
        t2 = perf_counter()
    else:
        resize = transforms.Resize((options.img_height, options.img_height))
        processed = resize(img)
        t2 = perf_counter()

    tensor = transforms.ToTensor()(processed)
    t3 = perf_counter()

    normalized = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )(tensor)
    t4 = perf_counter()

    return {
        'path': img_path,
        'disk_load': t1 - t0,
        'bit_patch' if options.isPatch else 'resize': t2 - t1,
        'to_tensor': t3 - t2,
        'normalize': t4 - t3,
        'total': t4 - t0,
    }


def print_report(timings, options):
    stage_keys = [k for k in timings[0].keys() if k != 'path']

    header = f"{'#':>4}  " + "  ".join(f"{k:>12}" for k in stage_keys) + "   path"
    print(header)
    print("-" * len(header))
    for i, t in enumerate(timings):
        row = f"{i:>4}  " + "  ".join(f"{t[k]*1000:>10.2f}ms" for k in stage_keys)
        print(f"{row}   {t['path']}")

    print("\n=== Summary over {} images ===".format(len(timings)))
    for k in stage_keys:
        values = [t[k] for t in timings]
        total = sum(values)
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        print(f"{k:>12}: mean={mean*1000:8.2f}ms  std={std*1000:7.2f}ms  "
              f"min={min(values)*1000:8.2f}ms  max={max(values)*1000:8.2f}ms  "
              f"total={total:7.3f}s")


def main():
    options = build_options()

    print(f"Dataset: {options.dataset_name}  isPatch={options.isPatch}  "
          f"bit_mode={options.bit_mode}  patch_mode={options.patch_mode}  "
          f"patch_size={options.patch_size}  img_height={options.img_height}  "
          f"unbiased={options.unbiased}\n")

    dataset = GenerativeImageTrainingSet(options.image_root, options.dataset_name, options)

    num_images = min(options.num_images, len(dataset))
    if num_images < options.num_images:
        print(f"Requested {options.num_images} images but dataset only has "
              f"{len(dataset)}, using {num_images}.")

    timings = []
    for i in range(num_images):
        timings.append(time_single_image(dataset, i, options))

    print_report(timings, options)


if __name__ == '__main__':
    main()
