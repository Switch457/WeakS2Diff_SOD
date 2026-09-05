import sys

import torch
from utils.train_utils import set_random_seed

from utils import init_env
import os
import argparse
from pathlib import Path

from utils.collate_utils import collate
from utils.import_utils import instantiate_from_config, recurse_instantiate_from_config, get_obj_from_str
from utils.init_utils import add_args
from torch.utils.data import DataLoader
from utils.trainer import Trainer

set_random_seed(7)


def get_loader(cfg):
    DUT_test_dataset = instantiate_from_config(cfg.test_dataset.DUT) 
    LFSD_test_dataset = instantiate_from_config(cfg.test_dataset.LFSD)
    NJU2K_test_dataset = instantiate_from_config(cfg.test_dataset.NJU2K)
    NLPR_test_dataset = instantiate_from_config(cfg.test_dataset.NLPR)
    SIP_test_dataset = instantiate_from_config(cfg.test_dataset.SIP)
    SSD_test_dataset = instantiate_from_config(cfg.test_dataset.SSD)
    STERE_test_dataset = instantiate_from_config(cfg.test_dataset.STERE)

    DUT_test_loader = DataLoader(
        DUT_test_dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate
    )
    LFSD_test_loader = DataLoader(
        LFSD_test_dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate
    )
    NJU2K_test_loader = DataLoader(
        NJU2K_test_dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate
    )
    NLPR_test_loader = DataLoader(
        NLPR_test_dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate
    )
    SIP_test_loader = DataLoader(
        SIP_test_dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate
    )
    SSD_test_loader = DataLoader(
        SSD_test_dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate
    )
    
    STERE_test_loader = DataLoader(
        STERE_test_dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate
    )

    return DUT_test_loader, LFSD_test_loader, NJU2K_test_loader, NLPR_test_loader, SIP_test_loader, SSD_test_loader, STERE_test_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--results_folder', type=str, default='./results')
    parser.add_argument('--num_epoch', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gradient_accumulate_every', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_sample_steps', type=int, default=None)
    parser.add_argument('--target_dataset', nargs='+', type=str, default= ['DUT-RGBD', 'LFSD', 'NJU2K','NLPR', 'SIP', 'SSD', 'STERE'])
    parser.add_argument('--time_ensemble', action='store_true')
    parser.add_argument('--batch_ensemble', action='store_true')

    cfg = add_args(parser)
    assert not (cfg.time_ensemble and cfg.batch_ensemble), 'Cannot use both time_ensemble and batch_ensemble'
    """
        Hack config here.
    """
    if cfg.num_sample_steps is not None:
        cfg.diffusion_model.params.num_sample_steps = cfg.num_sample_steps

    DUT_test_loader, LFSD_test_loader, NJU2K_test_loader, NLPR_test_loader, SIP_test_loader, SSD_test_loader, STERE_test_loader = get_loader(cfg)

    cond_uvit = instantiate_from_config(cfg.cond_uvit,
                                        conditioning_klass=get_obj_from_str(cfg.cond_uvit.params.conditioning_klass))
    model = recurse_instantiate_from_config(cfg.model,
                                            unet=cond_uvit)

    diffusion_model = instantiate_from_config(cfg.diffusion_model,
                                              model=model)

    optimizer = instantiate_from_config(cfg.optimizer, params=model.parameters())

    trainer = Trainer(
        diffusion_model,
        train_loader=None, test_loader=None,
        train_val_forward_fn=get_obj_from_str(cfg.train_val_forward_fn),
        gradient_accumulate_every=cfg.gradient_accumulate_every,
        results_folder=cfg.results_folder,
        optimizer=optimizer,
        train_num_epoch=cfg.num_epoch,
        amp=cfg.fp16,
        log_with=None,
        cfg=cfg,
    )

    trainer.load(pretrained_path=cfg.checkpoint)
    DUT_test_loader, LFSD_test_loader, NJU2K_test_loader, NLPR_test_loader, SIP_test_loader, SSD_test_loader, STERE_test_loader = \
        trainer.accelerator.prepare(DUT_test_loader, LFSD_test_loader, NJU2K_test_loader, NLPR_test_loader, SIP_test_loader, SSD_test_loader, STERE_test_loader)

    dataset_map = {
        'DUT-RGBD': DUT_test_loader,
        'LFSD': LFSD_test_loader,
        'NJU2K': NJU2K_test_loader,
        'NLPR': NLPR_test_loader,
        'SIP': SIP_test_loader,
        'SSD': SSD_test_loader,
        'STERE': STERE_test_loader ,
    }
    assert all([d_name in dataset_map.keys() for d_name in cfg.target_dataset]), \
        f'Invalid dataset name. Available dataset: {dataset_map.keys()}' \
        f'Your input: {cfg.target_dataset}'
    target_dataset = [(dataset_map[dataset_name], dataset_name) for dataset_name in cfg.target_dataset]

    for dataset, dataset_name in target_dataset:
        trainer.model.eval()
        mask_path = Path(cfg.test_dataset.STERE.params.image_root).parent.parent
        save_to = Path(cfg.results_folder) / dataset_name
        os.makedirs(save_to, exist_ok=True)
        if cfg.batch_ensemble:
            
            mae, _ = trainer.val_batch_ensemble(model=trainer.model,
                                                test_data_loader=dataset,
                                                accelerator=trainer.accelerator,
                                                thresholding=False,
                                                save_to=save_to)
        elif cfg.time_ensemble:
            print('time')
            mae, _ = trainer.val_time_ensemble(model=trainer.model,
                                               test_data_loader=dataset,
                                               accelerator=trainer.accelerator,
                                               thresholding=False,
                                               save_to=save_to)
        else:
            print('val')
            mae, _ = trainer.val(model=trainer.model,
                                 test_data_loader=dataset,
                                 accelerator=trainer.accelerator,
                                 thresholding=False,
                                 save_to=save_to)
        trainer.accelerator.wait_for_everyone()
        trainer.accelerator.print(f'{dataset_name} mae: {mae}')

        if trainer.accelerator.is_main_process:
            from utils.eval import eval

            eval_score = eval(
                mask_path=mask_path,
                pred_path=cfg.results_folder,
                dataset_name=dataset_name)
        trainer.accelerator.wait_for_everyone()
