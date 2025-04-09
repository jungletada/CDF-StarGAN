"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import cv2
import shutil
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import OrderedDict


from metrics.fid import calculate_fid_given_paths
from metrics.lpips import calculate_lpips_given_images
from core.data_loader_cfd import get_eval_loader
from core import utils


color_map ={'pressure':'viridis', 'temperature':'PiYG', 'velocity':'magma'}


def load_transform_image(file_path, target_height=256, target_width=512):
    img = Image.open(file_path).convert('L')
    image = np.array(img)
    original_height, original_width = image.shape
    scale = target_height / original_height
    new_width = int(original_width * scale)
    resized = cv2.resize(image, (new_width, target_height), interpolation=cv2.INTER_LINEAR)
        
    width_left = (resized.shape[1] - target_width) // 2
    cropped = resized[:, width_left:width_left + target_width]
    
    img = 1.0 - (cropped.astype(np.float32) / 255.0)
    # img = np.expand_dims(img, axis=0)  # 形状变为 (1, H, W)
    
    tensor = torch.from_numpy(img)
    return tensor


def visualize_results(contour, prediction, file_name, gt_name, cmap=None):
    # Create a plot with the reversed grayscale colormap so 0=white, 1=black
    if cmap is None:
        cmap = 'gray_r'
    pred = (prediction * contour).cpu()
    label = load_transform_image(gt_name)
    label = label * contour.cpu()
    data = torch.cat([pred, label], dim=0)
    data = data.numpy()

    plt.figure()
    plt.imshow(data, cmap=cmap, vmin=0, vmax=1)  # vmin/vmax set to 0-1 for proper color scaling
    plt.axis('off')  # Optional: turn off axes for a cleaner image
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    plt.close()


@torch.no_grad()
def calculate_metrics(nets, args, step, mode):
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
        if mode == 'reference':
            path_ref = os.path.join(args.train_img_dir, trg_domain)
            loader_ref = get_eval_loader(root=path_ref,
                                         img_size=args.img_size,
                                         batch_size=args.val_batch_size,
                                         imagenet_normalize=False,
                                         drop_last=True)

        for src_idx, src_domain in enumerate(src_domains):
            path_src = os.path.join(args.val_img_dir, src_domain)
            loader_src = get_eval_loader(root=path_src,
                                         img_size=args.img_size,
                                         batch_size=args.val_batch_size,
                                         imagenet_normalize=False)

            task = '%s-to-%s' % (src_domain, trg_domain)
            path_fake = os.path.join(args.eval_dir, task)
            shutil.rmtree(path_fake, ignore_errors=True)
            os.makedirs(path_fake)

            # lpips_values = []
            print('Generating images for %s...' % task)
            for i, input_dict in enumerate(tqdm(loader_src)):
                x_src = input_dict['image']
                filename = input_dict['filename']
                N = x_src.size(0)
                x_src = x_src.to(device)
                y_trg = torch.tensor([trg_idx] * N).to(device)
                masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None

                group_of_images = []
                for j in range(args.num_outs_per_domain):
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
                
                # save generated images to calculate FID later
                for k in range(N):
                    fname = filename[k]
                    pred_name = os.path.join(path_fake, fname.replace('.tiff', '.png'))
                    gt_name = os.path.join(args.val_img_dir, trg_domain, fname)
                    visualize_results(contour=x_src[k].squeeze(), 
                                      prediction=avg_image[k], 
                                      file_name=pred_name,
                                      gt_name=gt_name, 
                                      cmap=color_map[trg_domain])
                    
                #     utils.save_image(avg_image, ncol=1, filename=filename)
        ##############################################################################

        #         lpips_value = calculate_lpips_given_images(group_of_images)
        #         lpips_values.append(lpips_value)

        #     # calculate LPIPS for each task (e.g. cat2dog, dog2cat)
        #     lpips_mean = np.array(lpips_values).mean()
        #     lpips_dict['LPIPS_%s/%s' % (mode, task)] = lpips_mean

        # # delete dataloaders
        # del loader_src
        # if mode == 'reference':
        #     del loader_ref
        #     del iter_ref

    # # calculate the average LPIPS for all tasks
    # lpips_mean = 0
    # for _, value in lpips_dict.items():
    #     lpips_mean += value / len(lpips_dict)
    # lpips_dict['LPIPS_%s/mean' % mode] = lpips_mean

    # # report LPIPS values
    # filename = os.path.join(args.eval_dir, 'LPIPS_%.5i_%s.json' % (step, mode))
    # utils.save_json(lpips_dict, filename)

    # # calculate and report fid values
    # calculate_fid_for_all_tasks(args, domains, step=step, mode=mode)


def calculate_fid_for_all_tasks(args, domains, step, mode):
    print('Calculating FID for all tasks...')
    fid_values = OrderedDict()
    for trg_domain in domains:
        src_domains = [x for x in domains if x != trg_domain]

        for src_domain in src_domains:
            task = '%s2%s' % (src_domain, trg_domain)
            path_real = os.path.join(args.train_img_dir, trg_domain)
            path_fake = os.path.join(args.eval_dir, task)
            print('Calculating FID for %s...' % task)
            fid_value = calculate_fid_given_paths(
                paths=[path_real, path_fake],
                img_size=args.img_size,
                batch_size=args.val_batch_size)
            fid_values['FID_%s/%s' % (mode, task)] = fid_value

    # calculate the average FID for all tasks
    fid_mean = 0
    for _, value in fid_values.items():
        fid_mean += value / len(fid_values)
    fid_values['FID_%s/mean' % mode] = fid_mean

    # report FID values
    filename = os.path.join(args.eval_dir, 'FID_%.5i_%s.json' % (step, mode))
    utils.save_json(fid_values, filename)
