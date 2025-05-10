"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import logging
import shutil
from PIL import Image
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import r2_score, mean_absolute_percentage_error

from core import utils
from core.data_loader_cfd import get_eval_loader
from core.data_loader_cfd import (
    STAT_pressure,
    STAT_temperature,
    STAT_velocity
)


color_map ={
    'pressure':'viridis', 
    'temperature':'PiYG', 
    'velocity':'magma'}


def load_label(label_name):
    image = torch.from_numpy(np.array(Image.open(label_name).convert('L')))
    image = (image / 255.).clip(0, 1)
    return image


def load_scale(key):
    if key.__contains__('pressure'):
        return STAT_pressure
    elif key.__contains__('temperature'):
        return STAT_temperature
    elif key.__contains__('velocity'):
        return STAT_velocity
    else:
        raise NotImplementedError
    

def visualize_results(mask, pred, file_name, cmap='gray_r'):
    # Create a plot with the reversed grayscale colormap
    if len(mask.shape) == 4 or len(mask.shape) == 3:
        mask = mask.squeeze()
        pred = pred.squeeze()
        
    mask = mask.cpu().numpy()
    pred = pred.cpu().numpy()
    
    # colorize
    cm = matplotlib.colormaps[cmap]
    img_colored_np = cm(pred)
    img_colored_np[~mask] = 1
    img_colored_np = (img_colored_np * 255).astype(np.uint8)
    img_colored = Image.fromarray(img_colored_np)
    img_colored.save(file_name)
    
    
def evaluate(pred, label, key, mask, denormalize=True):
    if len(mask.shape) == 4 or len(mask.shape) == 3:
        mask = mask.squeeze()
        pred = pred.squeeze()
        
    mask = mask.cpu().numpy()
    pred = pred.cpu().numpy()
    
    if denormalize:
        stat = load_scale(key)
        pred = pred * (stat['max'] - stat['min']) + stat['min']
        label = label * (stat['max'] - stat['min']) + stat['min']

    pred *= mask
    label *= mask
    
    # Flatten arrays to 1D for metric calculations
    y_true = label.flatten()
    y_pred = pred.flatten()
    print(y_true, y_pred)
    # Compute metrics
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape,
    }
    
    
@torch.no_grad()
def calculate_metrics(nets, args, mode):
    print('Calculating evaluation metrics...')
    assert mode in ['latent', 'reference']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    domains = os.listdir(args.val_img_dir)
    domains.sort()
    num_domains = len(domains)
    print('Number of domains: %d' % num_domains)

    # lpips_dict = OrderedDict()
    for trg_idx, trg_domain in enumerate(domains):
        if trg_domain == 'contour':
            continue
        src_domains = ['contour']
        print(f'src_domains={src_domains}, and target domain={trg_domain}')

        path_ref = os.path.join(args.train_img_dir, trg_domain)
        loader_ref = get_eval_loader(
                            root=path_ref,
                            batch_size=1,
                            drop_last=True,)

        for src_idx, src_domain in enumerate(src_domains):
            path_src = os.path.join(args.val_img_dir, src_domain)
            loader_src = get_eval_loader(root=path_src, batch_size=1)

            task = '%s-to-%s' % (src_domain, trg_domain)
            path_fake = os.path.join(args.eval_dir, task)
            shutil.rmtree(path_fake, ignore_errors=True)
            os.makedirs(path_fake)

            # lpips_values = []
            print('Generating images for %s...' % task)
            for _, data_dict in enumerate(tqdm(loader_src)):
                x_src = data_dict['image']
                filename = data_dict['filename']

                N = x_src.size(0)
                x_src = x_src.to(device)
                y_trg = torch.tensor([trg_idx] * N).to(device)
                masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None

                group_of_images = []
                for _ in range(args.num_outs_per_domain):
                    if mode == 'latent':
                        z_trg = torch.randn(N, args.latent_dim).to(device)
                        s_trg = nets.mapping_network(z_trg, y_trg)
                    else:
                        try:
                            x_ref = next(iter_ref).to(device)
                        except:
                            iter_ref = iter(loader_ref)
                            x_ref = next(iter_ref)['image'].to(device)

                        if x_ref.size(0) > N:
                            x_ref = x_ref[:N]
                        s_trg = nets.style_encoder(x_ref, y_trg)

                    x_fake = nets.generator(x_src, s_trg, masks=masks)
                    group_of_images.append(x_fake)
                   
                group_of_images = torch.cat(group_of_images, dim=1)
                avg_image = torch.mean(group_of_images, dim=1)
                
                # save generated images to evaluate
                for k in range(N):
                    name = filename[k]
                    pred_name = os.path.join(path_fake, name)
                    label_name = os.path.join(args.val_img_dir, trg_domain, name)
                    label = load_label(label_name)
                    
                    mask = (x_src[k] >= 0.5)
                    pred = ((avg_image[k] + 1.) / 2.).clip(0, 1)   
                    
                    visualize_results(
                        mask=mask, 
                        pred=pred, 
                        file_name=pred_name,
                        cmap=color_map[trg_domain])
                    
                    res = evaluate(
                             pred=pred,
                             label=label,
                             key=trg_domain,
                             mask=mask)
                    print(res)
           