"""
debug_bit_patch.py

Load a single raw image (no Dataset/DataLoader) and run it through the same
patch-extraction/selection logic as bit_patch.bit_patch(), but with the
intermediate candidate patches and their texture scores exposed so you can
actually see why a given patch got picked.

`compute()` (the texture-score metric) is imported straight from bit_patch.py
so scoring matches production exactly; the surrounding extraction/selection
logic is reimplemented here only because bit_patch() itself doesn't expose
its intermediates.

Usage:
    python debug_bit_patch.py --dataset GLIDE --category ai --index 0
    python debug_bit_patch.py --dataset GLIDE --category nature --index 3 \
        --bit_mode thresholding --patch_size 32 --patch_mode max
"""
import argparse
import os

import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

from bit_patch import compute


def load_single_image(image_root, dataset_name, category, split="train", index=0):
    """Grab one raw image the same way GenerativeImage{Training,Validation}Set
    locates images, without building the full dataset."""
    img_dir = os.path.join(image_root, dataset_name, split, category)
    filenames = sorted(os.listdir(img_dir))
    img_path = os.path.join(img_dir, filenames[index])
    print(f"Loading: {img_path}")
    return Image.open(img_path).convert("RGB")


def debug_bit_patch(img, img_height, bit_mode, patch_size, patch_mode):
    """Same steps as bit_patch.bit_patch(), with candidates/scores kept around."""
    img_np = np.array(img)

    if bit_mode == "scaling":
        mask_low = 0x07
        red_low3 = ((img_np[:, :, 0] & mask_low) * (255 // 7)).astype(np.uint8)
        green_low3 = ((img_np[:, :, 1] & mask_low) * (255 // 7)).astype(np.uint8)
        blue_low3 = ((img_np[:, :, 2] & mask_low) * (255 // 7)).astype(np.uint8)
        combined_image = cv2.merge((red_low3, green_low3, blue_low3))
    elif bit_mode == "thresholding":
        combined_image = img_np  # currently a no-op passthrough, see bit_patch.py
    else:
        raise ValueError(f"Unsupported bit_mode: {bit_mode}")

    h, w, _ = combined_image.shape
    combined_image = Image.fromarray(combined_image)
    if min(h, w) < patch_size:
        combined_image = transforms.Resize((img_height, img_height))(combined_image)

    num_patch = (img_height // patch_size) ** 2
    crop = transforms.RandomCrop(patch_size)
    patch_list = [crop(combined_image) for _ in range(num_patch)]
    scores = [compute(p) for p in patch_list]

    order = np.argsort(scores)[::-1]  # descending
    if patch_mode == "max":
        selected_idx = int(order[0])
    elif patch_mode == "min":
        selected_idx = int(order[-1])
    else:
        selected_idx = int(np.random.randint(len(patch_list)))

    final_patch = cv2.resize(np.array(patch_list[selected_idx]), (img_height, img_height))

    print(f"{num_patch} candidate {patch_size}x{patch_size} patches | "
          f"score min={min(scores)} max={max(scores)} mean={np.mean(scores):.1f}")
    print(f"Selected patch #{selected_idx} (mode={patch_mode}) score={scores[selected_idx]}")

    return {
        "patch_list": patch_list,
        "scores": scores,
        "order": order,
        "selected_idx": selected_idx,
        "final_patch": final_patch,
    }


def plot_debug(img, result, num_show=8):
    patch_list, scores, order, selected_idx = (
        result["patch_list"], result["scores"], result["order"], result["selected_idx"]
    )
    num_show = min(num_show, len(patch_list))
    shown = list(order[:num_show])
    if selected_idx not in shown:
        shown[-1] = selected_idx

    cols = num_show + 1
    fig, axes = plt.subplots(1, cols, figsize=(3 * cols, 3))

    axes[0].imshow(img)
    axes[0].set_title("original")
    axes[0].axis("off")

    for ax, idx in zip(axes[1:], shown):
        ax.imshow(np.array(patch_list[idx]))
        marker = " <-" if idx == selected_idx else ""
        ax.set_title(f"#{idx} s={scores[idx]}{marker}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(3, 3))
    plt.imshow(result["final_patch"])
    plt.title("final model input")
    plt.axis("off")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image_root", default="../GenImage_root")
    parser.add_argument("--dataset", default="GLIDE")
    parser.add_argument("--category", default="ai", choices=["ai", "nature"])
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--img_height", type=int, default=256)
    parser.add_argument("--bit_mode", default="thresholding", choices=["scaling", "thresholding"])
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--patch_mode", default="max", choices=["max", "min", "random"])
    parser.add_argument("--num_show", type=int, default=8)
    args = parser.parse_args()

    img = load_single_image(args.image_root, args.dataset, args.category, args.split, args.index)
    result = debug_bit_patch(img, args.img_height, args.bit_mode, args.patch_size, args.patch_mode)
    plot_debug(img, result, args.num_show)


if __name__ == "__main__":
    main()
