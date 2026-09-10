import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os, sys
import io
from PIL import Image
import numpy as np
import cv2
import sys
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union
import pandas as pd
from time import time

MODEL_NAME_MAP = {
    0: 'BigGAN',
    1: 'Midjourney',
    2: 'Wukong',
    3: 'Stable_Diffusion_v1.4',
    4: 'Stable_Diffusion_v1.5',
    5: 'ADM',
    6: 'GLIDE',
    7: 'VQDM'
}

### add seed as a flag that is added to options
### seed - epoch - max - min - random - thresh - scaling
### image_seed_epoch_max_scaling.jpg

def _relative_image_path(img_path, image_root):
    ### Turn an absolute/relative image path into something safe to nest under the
    ### patch cache dir, keeping it recognizable (dataset/train/category/filename).
    try:
        rel = os.path.relpath(img_path, start=image_root)
        if not rel.startswith(os.pardir):
            return rel
    except ValueError:
        pass  # e.g. paths on different drives on Windows, or --unbiased paths
              # (they come from the metadata CSV and aren't under image_root)
    _, tail = os.path.splitdrive(os.path.normpath(img_path))
    # strip '..'/'.'/empty segments so they can't escape the cache dir or eat
    # into the seed/bit_mode/patch_mode components when joined below
    parts = [p for p in tail.split(os.sep) if p not in ('', os.curdir, os.pardir)]
    return os.path.join(*parts) if parts else os.path.basename(img_path)


def get_patch_cache_path(options, img_path):
    ### patches/seed_{seed}/{bit_mode}/{patch_mode}/epoch_{current_epoch}/<relative path>.png
    ### epoch is in there because bit_patch's crop is randomized -- caching without it
    ### would freeze every epoch onto whichever patch got sampled first.
    ### Always saved as PNG since bit_patch operates on low-order bits, which
    ### JPEG recompression would destroy.
    current_epoch = getattr(options, 'current_epoch', 0)
    rel = _relative_image_path(img_path, options.image_root)
    rel = os.path.splitext(rel)[0] + '.png'
    return os.path.join(
        'patches', f'seed_{options.seed}', options.bit_mode, options.patch_mode,
        f'epoch_{current_epoch}', rel
    )


def create_preprocessing_pipeline(options, img_path=None):
    ## try except
    ## resizing happens in this function already
    if options.isPatch:
        def patch_step(img):
            # bit_patch returns the raw patch_size x patch_size crop (uncached and
            # cached alike), so the resize to img_height always happens here, after
            # the patch is obtained -- whichever source it came from.
            if options.load_from_disk and img_path is not None:
                cache_path = get_patch_cache_path(options, img_path)
                if os.path.exists(cache_path):
                    patch = np.array(Image.open(cache_path).convert('RGB'))
                else:
                    patch = bit_patch_process(
                        img, options.img_height, options.bit_mode,
                        options.patch_size, options.patch_mode
                    )
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    try:
                        Image.fromarray(patch).save(cache_path)
                    except Exception as e:
                        print(f"Could not cache patch to {cache_path}: {str(e)}")
            else:
                patch = bit_patch_process(
                    img, options.img_height, options.bit_mode,
                    options.patch_size, options.patch_mode
                )

            return cv2.resize(patch, (options.img_height, options.img_height))

        transform_func = transforms.Lambda(patch_step)
    else:
        transform_func = transforms.Resize((options.img_height, options.img_height))

    return transforms.Compose([
        transform_func,
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def apply_preprocessing(image, options, img_path=None):
    pipeline = create_preprocessing_pipeline(options, img_path)
    return pipeline(image)


def retrieve_unbiased_dataset(dataset_name, mode):
    df = pd.read_csv("../UnbiasedGenImage/data/genimage_metadata.csv")
    df['path'] = df['path'].str.replace('GenImage', '../GenImage_root', regex=False)
    ## Dataset translation dict for filtering dataframe along with the upper and lower bounds for width and height
    ## Name, lower_bound_width, upper_bound_width, lower_bound_height, upper_bound_height
    dataset_translation_dict = {'BigGAN' : ('BigGAN', 110, 146, 110, 146),
                                'Midjourney' : ('Midjourney', 940, 1100, 940, 1100),
                                'Wukong' : ('wukong', 450, 550, 450, 550),
                                'Stable_Diffusion_v1.4' : ('stable_diffusion_v_1_4', 450, 550, 450, 550),
                                'Stable_Diffusion_v1.5' : ('stable_diffusion_v_1_5', 450, 550, 450, 550),
                                'ADM' : ('ADM', 230, 280, 230, 280),
                                'GLIDE' : ('glide', 230, 280, 230, 280),
                                'VQDM' : ('VQDM', 230, 280, 230, 280)}
    dataset_name_unb, lb_w, ub_w, lb_h, ub_h = dataset_translation_dict[dataset_name]
    ## make sure the path from metadata csv adheres to the directory naming convention of LOTA
    for key, values in dataset_translation_dict.items():
        df['path'] = df['path'].str.replace('root/' + values[0], 'root/' + key, regex=False)
    df_unbiased_natural = df[ (df["generator"] == "nature") & (df["width"] >= lb_w) & (df["height"] >= lb_h) & (df["width"] <= ub_w) & (df["height"] <= ub_h) & (df["compression_rate"] == 96) & (df['mode'] == mode)]
    df_unbiased_ai = df[ (df["generator"] == dataset_name_unb) & (df['mode'] == mode)]

    return df_unbiased_natural, df_unbiased_ai

class GenerativeImageTrainingSet(Dataset):
    ### het enige wat je hoeft aan te passen is de _load_images functie, die moet een lijst met image paths maken.
    def __init__(self, root_dir, dataset_name, options):
        super().__init__()
        self.options = options
        
        ## GenImage_root, Wukong, train
        self.base_path = os.path.join(root_dir, dataset_name, "train")

        if not self.options.unbiased:
            self.natural_images = self._load_images("nature")
            self.ai_images = self._load_images("ai")
        if self.options.unbiased:
            df_unbiased_natural, df_unbiased_ai = retrieve_unbiased_dataset(dataset_name, 'train')
            self.natural_images = df_unbiased_natural['path'].tolist()
            self.ai_images = df_unbiased_ai['path'].tolist()
            print("loaded metadata.csv")

            ### load df of this particular category
            ### filter out the train images
            ### create a list of all these paths
            #1024×1024 (MJ), 512×512 (SD4, SD5, Wukong), 256×256 (GLIDE, ADM, VQDM), and 128×128
        self.all_images = self.natural_images + self.ai_images
        self.labels = torch.cat([
            torch.ones(len(self.natural_images)),
            torch.zeros(len(self.ai_images))
        ])
        print(f"Dataset size: {len(self.natural_images)} (natural) {len(self.ai_images)} (AI)")

    def _load_images(self, category):
        #### based on a flag either load as is, or apply that preloading filtering function that we saw in the unbiased genimage github
        category_path = os.path.join(self.base_path, category)
        #return GenImage_root/Wukong/train/nature/n12144580_10998.JPEG
        return [os.path.join(category_path, f) for f in os.listdir(category_path)]

    def _compress_img(self, image, qf):
        outputIoStream = io.BytesIO()
        image.save(outputIoStream, "JPEG", quality=qf, optimice=True)
        outputIoStream.seek(0)
        return Image.open(outputIoStream)


    def _load_rgb(self, img_path, label):
        try:
            with open(img_path, 'rb') as f:
                img = Image.open(f).convert('RGB')
                if self.options.unbiased: ### If the unbiased flag is on, compress to the configured qf
                    if label:
                        img = self._compress_img(img, self.options.qf)
                return img
        except Exception as e:
            print(f"Image Loading Error {img_path}: {str(e)}")
            return Image.new('RGB', (256, 256), (0, 0, 0))

    def __getitem__(self, index):
        start = time()
        try:
            label = self.labels[index]
            img = self._load_rgb(self.all_images[index], label)
            
        except:
            prev_index = max(0, index - 1)
            label = self.labels[prev_index]
            img = self._load_rgb(self.all_images[prev_index], label)
            

        processed_img = apply_preprocessing(img, self.options, self.all_images[index])
        #print('Time to process img:', time() - start, self.all_images[index], label)
        return processed_img, label

    def __len__(self):
        return len(self.all_images)

class GenerativeImageValidationSet(Dataset):
    def __init__(self, root_dir, dataset_name, is_natural, options):
        super().__init__()
        self.options = options
        self.base_path = os.path.join(root_dir, dataset_name, "val")

        category = "nature" if is_natural else "ai"
        self.img_dir = os.path.join(self.base_path, category)
        if not self.options.unbiased:
            self.image_paths = [os.path.join(self.img_dir, f)
                                for f in os.listdir(self.img_dir)]
        elif self.options.unbiased:
            df_unbiased_natural, df_unbiased_ai = retrieve_unbiased_dataset(dataset_name, 'val')
            self.image_paths = df_unbiased_natural['path'].tolist() if is_natural else df_unbiased_ai['path'].tolist()

        self.labels = torch.ones(len(self.image_paths)) if is_natural else torch.zeros(len(self.image_paths))

    def _compress_img(self, image, qf):
        outputIoStream = io.BytesIO()
        image.save(outputIoStream, "JPEG", quality=qf, optimice=True)
        outputIoStream.seek(0)
        return Image.open(outputIoStream)

    def _load_rgb(self, img_path, label):
        try:
            with open(img_path, 'rb') as f:
                img = Image.open(f).convert('RGB')
                if self.options.unbiased:
                    if label:
                        img = self._compress_img(img, self.options.qf)
                return img
        except Exception as e:
            print(f"Val Image Loading Error {img_path}: {str(e)}")
            return Image.new('RGB', (256, 256), (0, 0, 0))

    def __getitem__(self, index):
        label = self.labels[index]
        img = self._load_rgb(self.image_paths[index], label)

        processed_img = apply_preprocessing(img, self.options, self.image_paths[index])

        return processed_img, label

    def __len__(self):
        return len(self.image_paths)

def create_validation_loader(options, dataset_name, is_natural):
    val_dataset = GenerativeImageValidationSet(
        options.image_root, dataset_name, is_natural, options
    )

    def collate_batch(batch):
        inputs = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])
        return inputs, labels

    return DataLoader(
        val_dataset,
        batch_size=options.val_batchsize,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_batch
    ), len(val_dataset)


def setup_validation_loaders(options):
    choices = options.choices
    loaders = []

    for idx, selected in enumerate(choices):
        if selected:
            loader_info = {}
            model_name = MODEL_NAME_MAP[idx]
            print(f"Val dataset: {model_name}")

            loader_info['name'] = model_name
            loader_info['val_ai_loader'], loader_info['ai_size'] = create_validation_loader(
                options, model_name, False
            )
            loader_info['val_nature_loader'], loader_info['nature_size'] = create_validation_loader(
                options, model_name, True
            )
            print(f"Dataset size: {loader_info['nature_size']} (natural) {loader_info['ai_size']} (AI)")
            loaders.append(loader_info)

    return loaders

def create_training_loader(options):
    choices = options.choices
    root_dir = options.image_root

    datasets = []

    dataset_config = [
        (0, "BigGAN"),
        (1, "Midjourney"),
        (2, "Wukong"),
        (3, "Stable_Diffusion_v1.4"),
        (4, "Stable_Diffusion_v1.5"),
        (5, "ADM"),
        (6, "GLIDE"),
        (7, "VQDM")
    ]

    for idx, folder_name in dataset_config:
        if choices[idx]:
            print(f"Train dataset: {MODEL_NAME_MAP[idx]}")
            dataset = GenerativeImageTrainingSet(
                root_dir, folder_name, options
            )
            datasets.append(dataset)
            
            
    combined_dataset = torch.utils.data.ConcatDataset(datasets)

    return DataLoader(
        combined_dataset,
        batch_size=options.batchsize,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

def get_loader(opt):
    return create_training_loader(opt)

def get_val_loader(opt):
    return setup_validation_loaders(opt)

def get_single_loader(opt, image_dir, is_real):
    return create_validation_loader(opt, image_dir, is_real)

from bit_patch import bit_patch as bit_patch_process
