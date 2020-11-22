import argparse
import logging
import torch
import os

from torch import optim, nn
from datetime import datetime
from torchvision.transforms import transforms
from torchsummary import summary

from models import get_model
from dataset import get_dataset
from trainer import Trainer
from utils.show_utils import show_net_structure


def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--model', '-a', type=str, metavar='model', default='XceptionUnet',
                       help='UNet/MyDenseUNet/ResNet34_UNet/UNet++/MyChannelUNet/Attention_UNet/SegNet/R2UNet/CENet/'
                            'FCN32s/FCN16s/FCN8s/XceptionUnet')
    parse.add_argument('--dataset', type=str, default='IDRiD',
                       help='liver/esophagus/dsb2018_cell/corneal/drive_eye/ISBI_cell/'
                            'kaggle_lung/our_large/our_min/TN-SCUI/IDRiD')
    parse.add_argument('--deepsupervision', type=bool, default=False)
    parse.add_argument("--action", type=str, help="train/test/train&test", default="train")
    parse.add_argument("--epoch", type=int, default=60, help="epoch number")
    parse.add_argument("--batch_size", type=int, default=4, help="batch size")
    parse.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parse.add_argument("--eval", type=bool, default=True, help="use eval or not")
    parse.add_argument("--weights", type=str, default=None,
                       help="load model weights. eg:checkpoints/dsb2018_cell_30.pth")
    parse.add_argument("--ckp", type=str, default='checkpoints', help="the path of model weight file")
    now = datetime.now().strftime('%Y_%m_%d-%H-%M')
    parse.add_argument("--tensorboard_dir", type=str,
                       default=f'runs/{parse.parse_args().model}_{parse.parse_args().dataset}_{now}',
                       help="tensorboard dir")
    parse.add_argument("--log_dir", type=str, default=None, help="log dir")
    parse.add_argument("--threshold", type=float, default=None)
    return parse.parse_args()


def get_logging():
    filename = None
    if args.log_dir:
        dirname = os.path.join(args.log_dir, args.model, str(args.batch_size), str(args.dataset), str(args.epoch))
        filename = dirname + '/log.log'
        if not os.path.exists(dirname):
            os.makedirs(dirname)
    logging.basicConfig(
        filename=filename,
        level=logging.INFO,
        # format='%(asctime)s\n%(levelname)s:%(message)s'
        format='%(levelname)s:%(message)s'
    )
    return logging


if __name__ == "__main__":
    x_transforms = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 3通道
        # transforms.Normalize([0.5], [0.5]) #单通道
    ])
    y_transforms = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor()
    ])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = get_args()
    logging = get_logging()

    train_loader, val_loader, test_loader = get_dataset(args.dataset, args.batch_size, x_transforms,
                                                                y_transforms)

    inputs = next(iter(train_loader))[1]
    model = get_model(args.model, device, 3, 4)
    # show_net_structure(model)
    # summary(model, tuple(inputs.shape[1:]))

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)

    trainer = Trainer(model, criterion, optimizer, scheduler, train_loader, val_loader, test_loader,
                              args, device, logging)

    if 'train' in args.action:
        trainer.train()
    if 'test' in args.action:
        trainer.test()
