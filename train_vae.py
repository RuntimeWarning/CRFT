import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import ast
import re
import torch
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.optim import AdamW
from configs.configs import str2bool
from core.datasets import get_datasets
from core.model.Autoencoder import get_model
from core.helpers.evaluation import Evaluation
from core.helpers.earlystopping import EarlyStopping
from core.model.Autoencoder import DiagonalGaussianDistribution



parser = argparse.ArgumentParser()
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--value_scale', type=float, default=90.0)
parser.add_argument('--max_epoches', type=int, default=101)
parser.add_argument('--save_dir', type=str, default='model_data')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--is_train', type=str2bool, default=True, help='Train when true, evaluate when false.')
parser.add_argument('--pretrained_model', type=str, default="")
parser.add_argument('--thresholds', type=str, default='[20.0, 30.0, 35.0, 40.0]')
parser.add_argument('--model_name', type=str, default='vae')
parser.add_argument('--datasets', type=str, default='shanghai')
parser.add_argument('--output_length', type=int, default=25)
args = parser.parse_args()
args.thresholds = [16, 74, 133, 160, 181, 219] if args.datasets.split("_")[0] == 'sevir' else ast.literal_eval(args.thresholds)
args.value_scale = 255.0 if args.datasets.split("_")[0]=='sevir' else 90.0


def train_wrapper(model):
    checkpoint_saver = EarlyStopping(verbose=True)
    start_epoch = 1
    if args.pretrained_model:
        model_stats = torch.load(args.pretrained_model, weights_only=True)
        model.load_state_dict(model_stats)
        start_epoch = re.findall(r'/(\d+)/', args.pretrained_model)
        start_epoch = int(start_epoch[0])+1 if start_epoch else 1
    train_input_handle = get_datasets(name=args.datasets, opt='train',
                                      batch_size=args.batch_size, 
                                      num_workers=args.num_workers, 
                                      shuffle=True)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    reconstruction_loss = nn.MSELoss()
    kl_loss_weight = 1e-6
    for epoch in range(start_epoch, args.max_epoches):
        total_loss = []
        train_pbar = tqdm(train_input_handle, total=len(train_input_handle))
        for img in train_pbar:
            optimizer.zero_grad()
            B, T, H, W = img.shape
            img = img.to(device)
            img = img.reshape(B*T, -1, H, W)
            img = img * 2. - 1
            img = torch.repeat_interleave(img, 3, dim=1)
            latents = model(img, fn='encode_moments')
            samples = model(latents, fn='sample')
            posterior = DiagonalGaussianDistribution(latents)
            img_d = model(samples, fn='decode')
            loss = reconstruction_loss(img_d, img) + torch.mean(posterior.kl()) * kl_loss_weight
            loss.backward()
            optimizer.step()
            train_pbar.set_description('epoch: {} train loss: {:.4f}'.format(epoch, loss.item()))
            total_loss.append(loss.item())
        mean_loss = np.mean(total_loss)
        model_dict = model.state_dict()
        checkpoint_saver(mean_loss, model_dict,
                         args.model_name+'_'+args.save_dir+'_'+args.datasets, epoch)


def test_wrapper(model):
    model_stats = torch.load(args.pretrained_model, weights_only=True)
    model.load_state_dict(model_stats)
    test_input_handle = get_datasets(name=args.datasets, opt='test', 
                                     batch_size=args.batch_size, 
                                     num_workers=args.num_workers, 
                                     shuffle=False)
    test_pbar = tqdm(test_input_handle, total=len(test_input_handle))
    evaluater = Evaluation(seq_len=args.output_length, value_scale=args.value_scale,
                           thresholds=args.thresholds)
    with torch.no_grad():
        for img in test_pbar:
            img = img.to(device)
            B, T, H, W = img.shape
            img = img.reshape(B*T, -1, H, W)
            img = torch.repeat_interleave(img, 3, dim=1)
            latents = model(img * 2 - 1, fn='encode_moments')
            latents = model(latents, fn='sample')
            image = model(latents, fn='decode')
            image = (image / 2 + 0.5).clamp(0.0, 1.0)
            image = image.reshape(B, T, -1, H, W).cpu().numpy()[:, :, 2]
            img = img.reshape(B, T, -1, H, W).cpu().numpy()[:, :, 0]
            if args.datasets == 'cikm':
                img = img[:,:,13:-14,13:-14]
                image = image[:,:,13:-14,13:-14]
            evaluater.update(img.swapaxes(1, 0), image.swapaxes(1, 0))
        evaluater.save('./')

if __name__ == '__main__':
    print('Initializing models')
    model = get_model('../vae_weights_cikm.pth', args.is_train)
    model = nn.DataParallel(model)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    total = sum(param.nelement() for param in model.parameters())
    print("Main Model Parameters: %.2fM" % (total/1e6))
    if args.is_train:
        train_wrapper(model)
    else:
        test_wrapper(model)
