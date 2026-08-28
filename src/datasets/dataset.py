import os
import random
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from pathlib import Path

import numpy as np
from src.datasets.realesrgan import RealESRGAN_degradation



class PairedSROnlineTxtDataset(torch.utils.data.Dataset):
    def __init__(self, split=None, args=None):
        super().__init__()

        self.args = args
        self.split = split
        # Updated live by the training loop (dataloader_num_workers=0 → same process,
        # so this mutation is visible to __getitem__ without any IPC) to drive the
        # optional domain-randomization strength annealing schedule.
        self.current_step = 0
        self.total_steps = getattr(args, "max_train_steps", 1000) if args is not None else 1000
        if split == 'train':
            self.lr_list, self.prompt_list, self.gt_list = [], [], []
            with open(args.dataset_txt_paths, "r") as f:
                for line in f:
                    lr_path, prompt = [x.strip() for x in line.split(",", 1)]
                    gt_path = lr_path.replace("/LR_polycorr/", "/HR/").replace("/LR/", "/HR/")
                    assert gt_path != lr_path

                    self.lr_list.append(lr_path)
                    self.prompt_list.append(prompt)
                    self.gt_list.append(gt_path)


        elif split == 'test':
            self.input_folder = os.path.join(args.dataset_test_folder, "LR")
            self.output_folder = os.path.join(args.dataset_test_folder, "HR")
            self.lr_list = []
            self.gt_list = []
            lr_names = os.listdir(os.path.join(self.input_folder))
            gt_names = os.listdir(os.path.join(self.output_folder))
            assert len(lr_names) == len(gt_names)
            for i in range(len(lr_names)):
                self.lr_list.append(os.path.join(self.input_folder, lr_names[i]))
                self.gt_list.append(os.path.join(self.output_folder,gt_names[i]))
            self.crop_preproc = transforms.Compose([
                transforms.RandomCrop((args.resolution_ori, args.resolution_ori)),
                transforms.Resize((args.resolution_tgt, args.resolution_tgt)),
            ])
            assert len(self.lr_list) == len(self.gt_list)

    def __len__(self):
        return len(self.gt_list)

    def _domain_randomize(self, lr01, gt01):
        """Non-label appearance randomization for zero-shot cross-domain robustness.

        Applies the SAME photometric transform (gain, contrast, gamma, brightness shift)
        to BOTH the LR input and the HR target, so the LR→HR restoration mapping is
        preserved while the global appearance distribution is widened. This prevents the
        model from latching onto the source domain's (e.g. BUSI) specific
        brightness/contrast/dynamic-range statistics, improving transfer to Cardiac/HC18.
        These are deliberately NOT degradation labels (they do not feed the router).

        lr01, gt01: tensors in [0,1]. Returns the transformed pair, still in [0,1].

        Two knobs trade in-domain accuracy against OOD robustness without disabling
        randomization outright:
          - domain_rand_prob:        apply on only a FRACTION of samples (like RandAugment's
                                      p), so most steps still optimize purely for in-domain
                                      pixel statistics.
          - domain_rand_anneal_frac: linearly decays `strength` to 0 over the last fraction
                                      of TOTAL training steps, so early training learns
                                      appearance-invariant features while late training
                                      fine-tunes on clean in-domain statistics.
        """
        prob = float(getattr(self.args, "domain_rand_prob", 1.0))
        if prob < 1.0 and random.random() >= prob:
            return lr01, gt01

        s = float(getattr(self.args, "domain_rand_strength", 0.2))
        anneal_frac = float(getattr(self.args, "domain_rand_anneal_frac", 0.0))
        if anneal_frac > 0:
            total = max(1, self.total_steps)
            anneal_start_step = (1.0 - anneal_frac) * total
            if self.current_step > anneal_start_step:
                progress = min(1.0, (self.current_step - anneal_start_step) / (anneal_frac * total))
                s = s * (1.0 - progress)
        if s <= 0:
            return lr01, gt01
        gain     = random.uniform(1.0 - s, 1.0 + s)          # multiplicative gain
        contrast = random.uniform(1.0 - s, 1.0 + s)          # contrast factor
        gamma    = float(np.exp(random.uniform(-s, s)))      # log-uniform gamma (dynamic range)
        bshift   = random.uniform(-0.5 * s, 0.5 * s)         # additive brightness / depth shift
        def _apply(x):
            x = F.adjust_gamma(x, gamma)
            x = F.adjust_contrast(x, contrast)
            x = F.adjust_brightness(x, gain)
            return (x + bshift).clamp(0.0, 1.0)
        return _apply(lr01), _apply(gt01)

    def __getitem__(self, idx):

        if self.split == 'train':
            lr_path  = self.lr_list[idx]           # LR always from main txt
            prompt   = self.prompt_list[idx]       # prompt always from main txt
            gt_path  = self.gt_list[idx]           # default GT
            lr_img = Image.open(lr_path).convert("RGB")
            gt_img = Image.open(gt_path).convert("RGB")
            lr01 = F.to_tensor(lr_img)
            gt01 = F.to_tensor(gt_img)
            if getattr(self.args, "domain_randomization", False):
                lr01, gt01 = self._domain_randomize(lr01, gt01)
            lr_t = F.normalize(lr01, mean=[0.5], std=[0.5])
            gt_t = F.normalize(gt01, mean=[0.5], std=[0.5])

            example = {}
            example["prompt"] = prompt
            example["neg_prompt"] = self.args.neg_prompt_csd
            example["null_prompt"] = ""
            example["output_pixel_values"] = gt_t
            example["conditioning_pixel_values"] = lr_t

            return example
            
        elif self.split == 'test':
            input_img = Image.open(self.lr_list[idx]).convert('RGB')
            output_img = Image.open(self.gt_list[idx]).convert('RGB')
            img_t = self.crop_preproc(input_img)
            output_t = self.crop_preproc(output_img)
            # input images scaled to -1, 1
            img_t = F.to_tensor(img_t)
            img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
            # output images scaled to -1,1
            output_t = F.to_tensor(output_t)
            output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

            example = {}
            example["neg_prompt"] = self.args.neg_prompt_csd
            example["null_prompt"] = ""
            example["output_pixel_values"] = output_t
            example["conditioning_pixel_values"] = img_t
            example["base_name"] = os.path.basename(self.lr_list[idx])

            return example
