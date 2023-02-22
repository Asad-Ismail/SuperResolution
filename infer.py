import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import cv2
import numpy as np
from tqdm import tqdm
import time



def resize_padding(img,new_h,new_w):
    old_image_height, old_image_width, channels = img.shape
    color = (0,0,0)
    result = np.full((new_h,new_w, channels), color, dtype=np.uint8)
    x_center = (new_w - old_image_width) // 2
    y_center = (new_h - old_image_height) // 2
    # copy img image into center of result image
    result[y_center:y_center+old_image_height, x_center:x_center+old_image_width] = img
    return result

def get_padded(imgs):
    maxh=-1
    maxw=-1
    for img in imgs:
        h,w,_=img.shape
        maxh=max(maxh,h)
        maxw=max(maxw,w)
    concat_img=None
    for i, img in enumerate(imgs):
        if concat_img is None:
            img = resize_padding(img,maxh,maxw)
            #correct_img = cv2.resize(correct_img, (maxw,maxh), interpolation = cv2.INTER_AREA)
            concat_img=img
        else:
            img = resize_padding(img,maxh,maxw)
            concat_img=np.concatenate([concat_img, img], axis=1)
    return concat_img

def pixelated_image(img):
    pixel_size = 8
    new_height = img.shape[0] * pixel_size
    new_width = img.shape[1] * pixel_size
    new_img = np.zeros((new_height, new_width, 3), np.uint8)
    for i in range(new_height):
        for j in range(new_width):
            new_img[i,j] = img[i//pixel_size,j//pixel_size]
    return new_img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_16_128.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_infer', action='store_true')
    
    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        wandb_logger = WandbLogger(opt)
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['val'], schedule_phase='val')
    
    logger.info('Begin Model Inference.')
    current_step = 0
    current_epoch = 0
    idx = 0

    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)
    execution_times = []
    for _,  val_data in tqdm(enumerate(val_loader)):
        idx += 1
        #print(f"Val loader keys are {val_data.keys()}")

        #Metrics.save_img(Metrics.tensor2img(val_data['HR']),f"{result_path}/{idx}_inp.png")
        #if idx not in [6526,18909,22113,38736,66136,113078,113082,125148,129344,131898,132614]:
        #    continue

        diffusion.feed_data(val_data)
        start_time = time.monotonic()
        diffusion.test(continous=True)
        end_time = time.monotonic()
        execution_times.append(end_time - start_time)
        visuals = diffusion.get_current_visuals(need_LR=True)

        hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
        fake_img = Metrics.tensor2img(visuals['INF'])  # uint8
        lr_img = Metrics.tensor2img(visuals['LR'])
        #print(f"Dict keys are {visuals.keys()}")

        sr_img_mode = 'concat'
        if sr_img_mode == 'single':
            # single img series
            sr_img = visuals['SR']  # uint8
            sample_num = sr_img.shape[0]
            for iter in range(0, sample_num):
                Metrics.save_img(
                    Metrics.tensor2img(sr_img[iter]), '{}/{}_{}_sr_{}.png'.format(result_path, current_step, idx, iter))
        elif sr_img_mode == 'grid':
            # grid img
            sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
            Metrics.save_img(sr_img, '{}/{}_{}_sr_process.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                Metrics.tensor2img(visuals['SR'][-1]), '{}/{}_{}_sr.png'.format(result_path, current_step, idx))
        else:
            # concatenation mode
            #print(f"fake image shape is {lr_img.shape}")
            logging.info(f"Saving images in {result_path}")
            h,w,_=Metrics.tensor2img(visuals['SR'][-1]).shape
            cv_upsample=cv2.resize(lr_img, dsize=(w,h),interpolation=cv2.INTER_CUBIC)
            Metrics.save_img(get_padded([lr_img,cv_upsample,Metrics.tensor2img(visuals['SR'][-1]),hr_img]),f"{result_path}/{idx}_concat.png")
        #Metrics.save_img(hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
        #Metrics.save_img(fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))

        if wandb_logger and opt['log_infer']:
            wandb_logger.log_eval_data(fake_img, Metrics.tensor2img(visuals['SR'][-1]), hr_img)

    if wandb_logger and opt['log_infer']:
        wandb_logger.log_eval_table(commit=True)

    average_time = sum(execution_times) / len(execution_times)
    # print the result
    print("Average execution time:", average_time)