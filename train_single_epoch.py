"""
train_single_epoch.py

Runs exactly one epoch (+ validation) of train.py's pipeline, reusing
train.py's own execute_training_iteration/perform_validation/etc. directly
so there's no logic drift between this and the real training script.

Why not just `python train.py --epoch 1`: train.py's poly_lr schedule uses
config.epoch as both the loop length *and* the schedule's max_iter, so
--epoch 1 makes curr_iter == max_iter == 1, and poly_lr's (1 - curr_iter /
max_iter) term becomes 0 -> the learning rate is 0 for the entire run (no
weights actually update). This script decouples "how many epochs I'm
nominally training for" (--epoch, keep it at your real total, e.g. 30) from
"which single epoch to run" (--target_epoch, default 1).

Output goes to its own directory, separate from config.save_path (train.py's
normal checkpoint location), so debug runs never clobber Network_best.pth /
config.json from a real, in-progress long-form training run. Default is
config.save_path with "../weights" swapped for "../weights_debug" and an
"epoch_<N>" subfolder appended; override with --output_dir. That directory
gets:
  - config.json                 (the run config, via toolkit.save_config)
  - Network_epoch_<N>.pth       (model weights after this epoch)
  - loss.png / loss.json        (per-iteration training loss for this epoch)
  - accuracy.png / accuracy.json (per-dataset + overall validation accuracy)

Caveat: train.py tracks training state (config, total_batches,
iteration_counter, best_performing_epoch, highest_accuracy) as module-level
globals rather than passing them around -- that's how main_execution() itself
works. This script pokes those same globals on the imported module before
calling its functions, for consistency with that design.

Usage:
    python train_single_epoch.py --choice 0 0 0 0 0 0 1 0 \\
        --image_root ../GenImage_root --bit_mode thresholding \\
        --patch_size 32 --patch_mode max --epoch 30 --target_epoch 1
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # safe on headless/remote boxes; plots are saved, not shown
import matplotlib.pyplot as plt
import torch

import util as toolkit
import train as _train  # noqa: E402  (reuses its functions as-is)


def pop_cli_value(argv, flag, default=None, cast=str):
    """Strip `flag value` out of argv before config.py's argparse sees it
    (config.py's parser is strict and would reject an unknown flag)."""
    if flag not in argv:
        return default
    i = argv.index(flag)
    value = cast(argv[i + 1])
    del argv[i:i + 2]
    return value


def make_debug_output_dir(save_path, target_epoch, override=None):
    if override:
        return override
    debug_root = save_path.replace("../weights", "../weights_debug", 1)
    return os.path.join(debug_root, f"epoch_{target_epoch}")


def plot_loss(batch_losses, output_dir, target_epoch):
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(batch_losses) + 1), batch_losses)
    plt.xlabel("iteration")
    plt.ylabel("BCE loss")
    plt.title(f"Training loss - epoch {target_epoch}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss.png"), dpi=150)
    plt.close()

    with open(os.path.join(output_dir, "loss.json"), "w") as f:
        json.dump({"target_epoch": target_epoch, "batch_losses": batch_losses}, f, indent=2)


def plot_accuracy(overall_accuracy, per_dataset_accuracy, output_dir, target_epoch):
    names = list(per_dataset_accuracy.keys()) + ["overall"]
    values = list(per_dataset_accuracy.values()) + [overall_accuracy]

    plt.figure(figsize=(max(6, len(names) * 1.2), 4))
    bars = plt.bar(names, values)
    bars[-1].set_color("black")
    plt.ylim(0, 1)
    plt.ylabel("accuracy")
    plt.title(f"Validation accuracy - epoch {target_epoch}")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy.png"), dpi=150)
    plt.close()

    with open(os.path.join(output_dir, "accuracy.json"), "w") as f:
        json.dump(
            {"target_epoch": target_epoch, "overall_accuracy": overall_accuracy,
             "per_dataset_accuracy": per_dataset_accuracy},
            f, indent=2,
        )


def main():
    target_epoch = pop_cli_value(sys.argv, "--target_epoch", default=1, cast=int)
    output_dir_override = pop_cli_value(sys.argv, "--output_dir", default=None, cast=str)

    torch.set_num_threads(2)
    toolkit.set_random_seed()

    config = _train.Configurator().parse()
    val_config = _train.prepare_validation_config()

    train_loader = _train.fetch_train_data(config)
    val_loader = _train.fetch_val_data(val_config)
    _train.configure_gpu(config.gpu_id)

    model = _train.NeuralNetwork().cuda()
    if config.load:
        model.load_state_dict(torch.load(config.load))
        print(f"Loaded model from {config.load}")

    optimizer = torch.optim.Adam(model.parameters(), config.lr)

    output_dir = make_debug_output_dir(config.save_path, target_epoch, output_dir_override)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Debug output directory: {output_dir}")

    config_path = toolkit.save_config(config, output_dir)
    print(f"Saved config to {config_path}")

    # Seed train.py's module-level state, matching main_execution()'s own init.
    _train.config = config
    _train.total_batches = len(train_loader)
    _train.iteration_counter = 0
    _train.best_performing_epoch = 0
    _train.highest_accuracy = 0

    current_lr = toolkit.poly_lr(optimizer, config.lr, target_epoch, config.epoch)
    print(f"Running epoch {target_epoch}/{config.epoch} | lr={current_lr:.6f}")

    batch_losses = _train.execute_training_iteration(
        train_loader, model, optimizer, target_epoch, output_dir
    )
    # execute_training_iteration only checkpoints every 50 epochs; always save
    # weights here since this is the only epoch this script runs.
    weights_path = os.path.join(output_dir, f"Network_epoch_{target_epoch}.pth")
    torch.save(model.state_dict(), weights_path)
    print(f"Saved weights to {weights_path}")

    overall_accuracy, per_dataset_accuracy = _train.perform_validation(
        val_loader, model, target_epoch, output_dir
    )

    plot_loss(batch_losses, output_dir, target_epoch)
    plot_accuracy(overall_accuracy, per_dataset_accuracy, output_dir, target_epoch)
    print(f"Saved loss.png / accuracy.png (+ .json) to {output_dir}")


if __name__ == '__main__':
    main()
