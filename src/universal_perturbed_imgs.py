import pyrootutils
import copy
from tqdm import tqdm
import numpy as np

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

# ------------------------------------------------------------------------------------ #
# `pyrootutils.setup_root(...)` is an optional line at the top of each entry file
# that helps to make the environment more robust and convenient
#
# the main advantages are:
# - allows you to keep all entry files in "src/" without installing project as a package
# - makes paths and scripts always work no matter where is your current work dir
# - automatically loads environment variables from ".env" file if exists
#
# how it works:
# - the line above recursively searches for either ".git" or "pyproject.toml" in present
#   and parent dirs, to determine the project root dir
# - adds root dir to the PYTHONPATH (if `pythonpath=True`), so this file can be run from
#   any place without installing project as a package
# - sets PROJECT_ROOT environment variable which is used in "configs/paths/default.yaml"
#   to make all paths always relative to the project root
# - loads environment variables from ".env" file in root dir (if `dotenv=True`)
#
# you can remove `pyrootutils.setup_root(...)` if you:
# 1. either install project as a package or move each entry file to the project root dir
# 2. simply remove PROJECT_ROOT variable from paths in "configs/paths/default.yaml"
# 3. always run entry files from the project root dir
#
# https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

import os
from typing import List, Tuple

import hydra
import pytorch_lightning as pl
import torch
import torchvision
import torchvision.transforms.functional as TF
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers.logger import Logger

from src import utils

torch.set_float32_matmul_precision("medium")

log = utils.get_pylogger(__name__)

# def grads(model, img, target_mask):
#     model.eval()
#     image_inp = image_inp.clone().detach().requires_grad_(True)

#     output = model(image_inp)  # [B, C, H, W]

#     if isinstance(target_class, int):
#         # select fixed class channel across the whole image
#         selected = output[:, target_class, :, :]  # [B, H, W]
#     else:
#         # If per-pixel class targets are provided
#         selected = output.gather(1, target_class.unsqueeze(1))  # [B, 1, H, W]
#         selected = selected.squeeze(1)  # [B, H, W]

#     selected.sum().backward()

#     return image_inp.grad  # [B, 3, H, W]
    

@utils.task_wrapper
def gen_perturbed_imgs(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    if cfg.use_ckpt:
        assert cfg.ckpt_path, "You must provide a checkpoint path when `use_ckpt=True`"

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    
    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
    }
    train_loader = datamodule.train_dataloader()
    if cfg.use_ckpt:
        log.info(f"Loading checkpoint from <{cfg.ckpt_path}>")
        checkpoint = torch.load(cfg.ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
        
    if cfg.trainer.accelerator == "gpu":
        if cfg.trainer.devices:
            device = torch.device(f"cuda:{cfg.trainer.devices[0]}")
        else:
            device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model.to(device)
    
    if not os.path.exists(cfg["adv_imgs_dir"]):
        os.makedirs(cfg["adv_imgs_dir"])
    # v = 0 # This is universal perturbation. Same as the image shape 
    v = torch.zeros_like(torch.empty(1,1,352,352)).to(device)
    fooling_rate = 0.0
    for batch in tqdm(train_loader):
        print('Inisdeeeeeeeeeee')

        batch["pixel_values"] = batch["pixel_values"].to(device) # torch.Size([1, 3, 352, 352])
        batch["mask"] = batch["mask"].to(device) # Ground truth
        batch["attention_mask"] = batch["attention_mask"].to(device)
        batch["input_ids"] = batch["input_ids"].to(device)
        
        print(batch['pixel_values'].shape)
        
        with torch.no_grad():
            print('No gradddddddd')
            perturbed_batch = batch
            perturbed_batch['pixel_values'] = batch['pixel_values'] + v
            output1 = model.step(batch)
            predicted_mask= output1['preds']
            perturbed_img_output =   model.step(perturbed_batch)
            perturbed_img_output_mask =  perturbed_img_output['preds']
            print('Done', predicted_mask.shape, perturbed_img_output_mask.shape)
        mask1 = torch.argmax(predicted_mask, dim=1)
        mask2 = torch.argmax(perturbed_img_output_mask, dim=1)
        
        print('========mask1====', mask1.shape)
        print('========mask2====', mask2.shape)
        
        pixel_diff_ratio = (mask1 != mask2).float().mean()
        print(pixel_diff_ratio, '================')
        exit()
        
        if pixel_diff_ratio <= 0.2:
            f_img =   model.step(perturbed_batch)
            mask = perturbed_batch['preds']
            
            img_pixel_shape = perturbed_batch['pixel_values'].shape
            pert_img = perturbed_batch
            
            f_i = torch.argmax(perturbed_batch['preds'], dim=1)
            # k_i = 
            
            w = torch.zeros(img_pixel_shape)
            r_tot = np.zeros(img_pixel_shape)
            
            loop_i = 0
            
            while loop_i < 50:
                pert = torch.inf
                # gradients = 
                # gradients = grads()
            
             
            

        # batch["pixel_values"].requires_grad = True
        

        
        step_out = model.step(batch)
        predicted_mask = step_out["preds"]
        print(predicted_mask.shape, batch['mask'].shape)
        perturbed_batch = batch['pixel_values'] + v
        print(batch['pixel_values'].shape, perturbed_batch['pixel_values'].shape)
        exit()
        
        
        # loss = step_out["loss"]
        # model.zero_grad()
        # loss.backward()
        # grad_signs = torch.sign(batch["pixel_values"].grad)
        # pert_imgs = batch["pixel_values"] + 0.1 * grad_signs
        # pert_imgs = denormalize(pert_imgs, mean=cfg["img_mean"], std=cfg["img_std"])

    return {}, object_dict
    
    

    # object_dict = {
    #     "cfg": cfg,
    #     "datamodule": datamodule,
    #     "model": model,
       
    # }
    # train_loader = datamodule.train_dataloader() # We are training for finding universal perturbation.
    # train_dataset = train_loader.dataset
    # total_samples = len(train_dataset)
    # print(f"Total training samples: {total_samples}") # 4800 for camus
    
    log.info("Generating perturbed images...")
    # Validate the existence of the directory to save the perturbed images
    if not os.path.exists(cfg["adv_imgs_dir"]):
        os.makedirs(cfg["adv_imgs_dir"])
        
    # Hyperparameters
    epsilon = 8/255
    alpha = 2/255 
    num_iterations = 10
    delta=0.2
    max_iter_uni = torch.inf
    p=torch.inf, 
    xi=10, 
    p=torch.inf, 
    num_classes=10, # Our involves masking, we need to manage this.
    overshoot=0.02, 
    max_iter_df=10
    
    v = 0 # This is universal perturbation. Same as the image shape 
    fooling_rate = 0.0
    while fooling_rate < 1 - delta:
        # Going through the dataset and compute the perturbation increments sequentially. 
        # for batch in tqdm(train_loader):  # We need to find universal perturbation that works for all images.
        #     print(batch)
        #     exit()
        
        
        
        
        pred_outputs = trainer.predict(
                model=model,
                dataloaders=datamodule,
                ckpt_path=cfg.ckpt_path,
            )

        preds, mask_names, heights, widths = [], [], [], []
        for p in pred_outputs:
            print(p["preds"].shape)
            exit()
            preds += list(p["preds"])
            mask_names += list(p["mask_names"])
            heights += list(p["heights"])
            widths += list(p["widths"])
            
        
    
    
    
    
    return {}, object_dict

def denormalize(
    image: torch.Tensor, mean: List[float], std: List[float]
) -> torch.Tensor:
    dtype = image.dtype
    device = image.device
    mean = torch.as_tensor(mean, dtype=dtype, device=device)
    std = torch.as_tensor(std, dtype=dtype, device=device)
    return image.mul_(std[:, None, None]).add_(mean[:, None, None])


@hydra.main(version_base="1.2", config_path="../configs", config_name="adv.yaml")
def main(cfg: DictConfig) -> None:
    gen_perturbed_imgs(cfg)


if __name__ == "__main__":
    main()
