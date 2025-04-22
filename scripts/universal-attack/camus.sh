#!/bin/bash

exp=clipseg_dense_adapter_vl
datamodule=camus
exp_name=${exp}_${datamodule}

python src/universal_perturbed_imgs.py \
    experiment=${exp} \
    experiment_name=${exp_name} \
    ckpt_path=logs/train/runs/${exp_name}/checkpoints/best.ckpt \
    datamodule=${datamodule}.yaml \
    datamodule.batch_size=1 \
    trainer.accelerator=gpu \
    trainer.precision=16-mixed \
    trainer.devices=[0] \
    prompt_type=random \
    seed=0 \
    logger=wandb.yaml \
    adv_imgs_dir=adv_imgs/universal/${exp_name}

python src/eval.py \
    experiment=${exp} \
    experiment_name=${exp_name} \
    ckpt_path=logs/train/runs/${exp_name}/checkpoints/best.ckpt \
    datamodule=${datamodule}.yaml \
    datamodule.batch_size=1 \
    trainer.accelerator=gpu \
    trainer.precision=16-mixed \
    trainer.devices=[0] \
    prompt_type=random \
    seed=0 \
    logger=wandb.yaml \
    task_name=eval \
    datamodule.test_dataset.images_dir=adv_imgs/universal/${exp_name}