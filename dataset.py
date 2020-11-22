import torch
import torch.utils.data as data
import PIL.Image as Image
from sklearn.model_selection import train_test_split
import os
import random
import numpy as np
from skimage.io import imread
import cv2
from glob import glob
import imageio
from torch.utils.data import DataLoader
from random import shuffle


class LiverDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.train_root = r"dataset\liver\train"
        self.val_root = r"dataset\liver\val"
        self.test_root = self.val_root
        self.pics, self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            root = self.train_root
        if self.state == 'val':
            root = self.val_root
        if self.state == 'test':
            root = self.test_root

        pics = []
        masks = []
        n = len(os.listdir(root)) // 2  # 因为数据集中一套训练数据包含有训练图和mask图，所以要除2
        for i in range(n):
            img = os.path.join(root, "%03d.png" % i)  # liver is %03d
            mask = os.path.join(root, "%03d_mask.png" % i)
            pics.append(img)
            masks.append(mask)
            # imgs.append((img, mask))
        return pics, masks

    def __getitem__(self, index):
        # x_path, y_path = self.imgs[index]
        x_path = self.pics[index]
        y_path = self.masks[index]
        origin_x = Image.open(x_path)
        origin_y = Image.open(y_path)
        # origin_x = cv2.imread(x_path)
        # origin_y = cv2.imread(y_path,cv2.COLOR_BGR2GRAY)
        if self.transform is not None:
            img_x = self.transform(origin_x)
        if self.target_transform is not None:
            img_y = self.target_transform(origin_y)
        return img_x, img_y, x_path, y_path

    def __len__(self):
        return len(self.pics)


class esophagusDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.train_root = r"dataset\esophagus\train"
        self.val_root = r"dataset\esophagus\test"
        self.test_root = self.val_root
        self.pics, self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            root = self.train_root
        if self.state == 'val':
            root = self.val_root
        if self.state == 'test':
            root = self.test_root
        pics = []
        masks = []
        n = len(os.listdir(root)) // 2  # 因为数据集中一套训练数据包含有训练图和mask图，所以要除2
        for i in range(n):
            img = os.path.join(root, "%05d.png" % i)  # liver is %03d
            mask = os.path.join(root, "%05d_mask.png" % i)
            pics.append(img)
            masks.append(mask)
            # imgs.append((img, mask))
        return pics, masks

    def __getitem__(self, index):
        # x_path, y_path = self.imgs[index]
        x_path = self.pics[index]
        y_path = self.masks[index]
        # origin_x = Image.open(x_path)
        # origin_y = Image.open(y_path)
        origin_x = cv2.imread(x_path)
        origin_y = cv2.imread(y_path, cv2.COLOR_BGR2GRAY)
        if self.transform is not None:
            img_x = self.transform(origin_x)
        if self.target_transform is not None:
            img_y = self.target_transform(origin_y)
        return img_x, img_y, x_path, y_path

    def __len__(self):
        return len(self.pics)


class dsb2018CellDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = True
        self.root = r'dataset\dsb2018_cell'
        self.img_paths = None
        self.mask_paths = None
        self.train_img_paths, self.val_img_paths = None, None
        self.train_mask_paths, self.val_mask_paths = None, None
        self.pics, self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        self.img_paths = glob(self.root + '\images\*')
        self.mask_paths = glob(self.root + '\masks\*')
        self.train_img_paths, self.val_img_paths, self.train_mask_paths, self.val_mask_paths = \
            train_test_split(self.img_paths, self.mask_paths, test_size=0.2, random_state=41)
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            return self.train_img_paths, self.train_mask_paths
        if self.state == 'val':
            return self.val_img_paths, self.val_mask_paths
        if self.state == 'test':
            return self.val_img_paths, self.val_mask_paths  # 因数据集没有测试集，所以用验证集代替

    def __getitem__(self, index):
        pic_path = self.pics[index]
        mask_path = self.masks[index]
        # origin_x = Image.open(x_path)
        # origin_y = Image.open(y_path)
        pic = Image.open(pic_path)
        mask = Image.open(mask_path)
        # if self.aug:
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[:, ::-1, :].copy()
        #         mask = mask[:, ::-1].copy()
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[::-1, :, :].copy()
        #         mask = mask[::-1, :].copy()
        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(mask)
        return img_x, img_y, pic_path, mask_path

    def __len__(self):
        return len(self.pics)


class CornealDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = True
        self.root = r'dataset\corneal'
        self.img_paths = None
        self.mask_paths = None
        self.train_img_paths, self.val_img_paths, self.test_img_paths = None, None, None
        self.train_mask_paths, self.val_mask_paths, self.test_mask_paths = None, None, None
        self.pics, self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        self.train_img_paths = glob(self.root + r'\training\train_images\*')
        self.train_mask_paths = glob(self.root + r'\training\train_mask\*')
        self.val_img_paths = glob(self.root + r'\val\val_images\*')
        self.val_mask_paths = glob(self.root + r'\val\val_mask\*')
        self.test_img_paths = glob(self.root + r'\test\test_images\*')
        self.test_mask_paths = glob(self.root + r'\test\test_mask\*')
        # self.train_img_paths, self.val_img_paths, self.train_mask_paths, self.val_mask_paths = \
        #     train_test_split(self.img_paths, self.mask_paths, test_size=0.2, random_state=41)
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            return self.train_img_paths, self.train_mask_paths
        if self.state == 'val':
            return self.val_img_paths, self.val_mask_paths
        if self.state == 'test':
            return self.test_img_paths, self.test_mask_paths

    def __getitem__(self, index):
        pic_path = self.pics[index]
        mask_path = self.masks[index]
        # origin_x = Image.open(x_path)
        # origin_y = Image.open(y_path)
        pic = Image.open(pic_path)
        mask = Image.open(mask_path)
        # if self.aug:
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[:, ::-1, :].copy()
        #         mask = mask[:, ::-1].copy()
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[::-1, :, :].copy()
        #         mask = mask[::-1, :].copy()
        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(mask)
        return img_x, img_y, pic_path, mask_path

    def __len__(self):
        return len(self.pics)


class DriveEyeDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = True
        self.root = r'dataset\drive_eye'
        self.pics, self.masks = self.getDataPath()
        self.img_paths = None
        self.mask_paths = None
        self.train_img_paths, self.val_img_paths, self.test_img_paths = None, None, None
        self.train_mask_paths, self.val_mask_paths, self.test_mask_paths = None, None, None
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        self.train_img_paths = glob(self.root + r'\training\images\*')
        self.train_mask_paths = glob(self.root + r'\training\1st_manual\*')
        self.val_img_paths = glob(self.root + r'\test\images\*')
        self.val_mask_paths = glob(self.root + r'\test\1st_manual\*')
        self.test_img_paths = self.val_img_paths
        self.test_mask_paths = self.val_mask_paths
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            return self.train_img_paths, self.train_mask_paths
        if self.state == 'val':
            return self.val_img_paths, self.val_mask_paths
        if self.state == 'test':
            return self.test_img_paths, self.test_mask_paths

    def __getitem__(self, index):
        imgx, imgy = (576, 576)
        pic_path = self.pics[index]
        mask_path = self.masks[index]
        # origin_x = Image.open(x_path)
        # origin_y = Image.open(y_path)
        # print(pic_path)
        pic = cv2.imread(pic_path)
        mask = cv2.imread(mask_path, cv2.COLOR_BGR2GRAY)
        if mask == None:
            mask = imageio.mimread(mask_path)
            mask = np.array(mask)[0]
        pic = cv2.resize(pic, (imgx, imgy))
        mask = cv2.resize(mask, (imgx, imgy))
        pic = pic.astype('float32') / 255
        mask = mask.astype('float32') / 255
        # if self.aug:
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[:, ::-1, :].copy()
        #         mask = mask[:, ::-1].copy()
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[::-1, :, :].copy()
        #         mask = mask[::-1, :].copy()
        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(mask)
        return img_x, img_y, pic_path, mask_path

    def __len__(self):
        return len(self.pics)


class IsbiCellDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = True
        self.root = r'dataset\ISBI_cell'
        self.img_paths = None
        self.mask_paths = None
        self.train_img_paths, self.val_img_paths, self.test_img_paths = None, None, None
        self.train_mask_paths, self.val_mask_paths, self.test_mask_paths = None, None, None
        self.pics, self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        self.img_paths = glob(self.root + r'\train\images\*')
        self.mask_paths = glob(self.root + r'\train\label\*')
        self.train_img_paths, self.val_img_paths, self.train_mask_paths, self.val_mask_paths = \
            train_test_split(self.img_paths, self.mask_paths, test_size=0.2, random_state=41)
        self.test_img_paths, self.test_mask_paths = self.val_img_paths, self.val_mask_paths
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            return self.train_img_paths, self.train_mask_paths
        if self.state == 'val':
            return self.val_img_paths, self.val_mask_paths
        if self.state == 'test':
            return self.test_img_paths, self.test_mask_paths

    def __getitem__(self, index):
        pic_path = self.pics[index]
        mask_path = self.masks[index]
        pic = Image.open(pic_path)
        mask = Image.open(mask_path)
        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(mask)
        return img_x, img_y, pic_path, mask_path

    def __len__(self):
        return len(self.pics)


class LungKaggleDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = True
        self.root = r'dataset\kaggle_lung'
        self.img_paths = None
        self.mask_paths = None
        self.train_img_paths, self.val_img_paths, self.test_img_paths = None, None, None
        self.train_mask_paths, self.val_mask_paths, self.test_mask_paths = None, None, None
        self.pics, self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        self.img_paths = glob(self.root + r'\images\*')
        self.mask_paths = glob(self.root + r'\masks\*')
        self.train_img_paths, self.val_img_paths, self.train_mask_paths, self.val_mask_paths = \
            train_test_split(self.img_paths, self.mask_paths, test_size=0.2, random_state=41)
        self.test_img_paths, self.test_mask_paths = self.val_img_paths, self.val_mask_paths
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            return self.train_img_paths, self.train_mask_paths
        if self.state == 'val':
            return self.val_img_paths, self.val_mask_paths
        if self.state == 'test':
            return self.test_img_paths, self.test_mask_paths

    def __getitem__(self, index):
        pic_path = self.pics[index]
        mask_path = self.masks[index]
        # origin_x = Image.open(x_path)
        # origin_y = Image.open(y_path)
        pic = Image.open(pic_path)
        mask = Image.open(mask_path)
        # if self.aug:
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[:, ::-1, :].copy()
        #         mask = mask[:, ::-1].copy()
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[::-1, :, :].copy()
        #         mask = mask[::-1, :].copy()
        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(mask)
        return img_x, img_y, pic_path, mask_path

    def __len__(self):
        return len(self.pics)


class OurLargeDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = True
        self.root = r'dataset\our_large'
        self.img_paths = None
        self.mask_paths = None
        self.train_img_paths, self.val_img_paths, self.test_img_paths = None, None, None
        self.train_mask_paths, self.val_mask_paths, self.test_mask_paths = None, None, None
        self.pics, self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        self.img_paths = glob(self.root + r'\images\*')
        self.mask_paths = glob(self.root + r'\masks\*')
        self.train_img_paths, self.val_img_paths, self.train_mask_paths, self.val_mask_paths = \
            train_test_split(self.img_paths, self.mask_paths, test_size=0.2, random_state=41)
        self.test_img_paths, self.test_mask_paths = self.val_img_paths, self.val_mask_paths
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            return self.train_img_paths, self.train_mask_paths
        if self.state == 'val':
            return self.val_img_paths, self.val_mask_paths
        if self.state == 'test':
            return self.test_img_paths, self.test_mask_paths

    def __getitem__(self, index):
        pic_path = self.pics[index]
        mask_path = self.masks[index]
        pic = Image.open(pic_path).convert('RGB')
        mask = Image.open(mask_path)
        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(mask)
        return img_x, img_y, pic_path, mask_path

    def __len__(self):
        return len(self.pics)


class OurMinDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = True
        self.root = r'dataset\our_min'
        self.img_paths = None
        self.mask_paths = None
        self.train_img_paths, self.val_img_paths, self.test_img_paths = None, None, None
        self.train_mask_paths, self.val_mask_paths, self.test_mask_paths = None, None, None
        self.pics, self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        self.img_paths = glob(self.root + r'\images\*')
        self.mask_paths = glob(self.root + r'\masks\*')
        self.train_img_paths, self.val_img_paths, self.train_mask_paths, self.val_mask_paths = \
            train_test_split(self.img_paths, self.mask_paths, test_size=0.2, random_state=41)
        self.test_img_paths, self.test_mask_paths = self.val_img_paths, self.val_mask_paths
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            return self.train_img_paths, self.train_mask_paths
        if self.state == 'val':
            return self.val_img_paths, self.val_mask_paths
        if self.state == 'test':
            return self.test_img_paths, self.test_mask_paths

    def __getitem__(self, index):
        pic_path = self.pics[index]
        mask_path = self.masks[index]
        pic = Image.open(pic_path).convert('RGB')
        mask = Image.open(mask_path)
        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(mask)
        return img_x, img_y, pic_path, mask_path

    def __len__(self):
        return len(self.pics)


class TNSCUIDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = True
        self.root = r'dataset\TN-SCUI'
        self.img_paths = None
        self.mask_paths = None
        self.train_img_paths, self.val_img_paths, self.test_img_paths = None, None, None
        self.train_mask_paths, self.val_mask_paths, self.test_mask_paths = None, None, None
        self.pics, self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        self.img_paths = glob(self.root + r'\images\*')
        self.mask_paths = glob(self.root + r'\masks\*')
        self.train_img_paths, self.val_img_paths, self.train_mask_paths, self.val_mask_paths = \
            train_test_split(self.img_paths, self.mask_paths, test_size=0.2, random_state=41)
        self.test_img_paths, self.test_mask_paths = self.val_img_paths, self.val_mask_paths
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            return self.train_img_paths, self.train_mask_paths
        if self.state == 'val':
            return self.val_img_paths, self.val_mask_paths
        if self.state == 'test':
            return self.test_img_paths, self.test_mask_paths

    def __getitem__(self, index):
        pic_path = self.pics[index]
        mask_path = self.masks[index]
        pic = Image.open(pic_path).convert('RGB')
        mask = Image.open(mask_path)

        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(mask)
        return img_x, img_y, pic_path, mask_path

    def __len__(self):
        return len(self.pics)


class IDRiDDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.suffixs = ['_MA','_HE','_EX','_SE']
        self.state = state
        self.aug = True
        self.root = r'dataset\IDRiD\segmentation'
        self.img_paths = []
        self.train_img_paths, self.val_img_paths, self.test_img_paths = None, None, None
        self.imgs = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        img_paths = os.listdir(os.path.join(self.root, 'images'))
        mask_paths = os.listdir(os.path.join(self.root, 'masks'))
        for img_path in img_paths:
            exist = True
            for suffix in self.suffixs:
                t = img_path[:8] + suffix + '.tif'
                if t not in mask_paths:
                    exist = False
                    break
            if exist:
                self.img_paths.append(img_path)
        shuffle(self.img_paths)
        self.train_img_paths, self.val_img_paths = train_test_split(self.img_paths, test_size=0.2, random_state=41)
        self.test_img_paths = self.val_img_paths
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            return self.train_img_paths
        if self.state == 'val':
            return self.val_img_paths
        if self.state == 'test':
            return self.test_img_paths

    def __getitem__(self, index):
        img_name = self.imgs[index]
        img = Image.open(os.path.join(os.path.join(self.root, 'images'),img_name)).convert('RGB')
        masks = []
        for suffix in self.suffixs:
            mask_name = img_name[:8] + suffix + '.tif'
            mask = Image.open(os.path.join(os.path.join(self.root, 'masks'), mask_name))
            if self.target_transform is not None:
                mask = self.target_transform(mask)
            masks.append(mask)
        mask = torch.cat(masks, dim=0)
        if self.transform is not None:
            img = self.transform(img)
        return img, mask

    def __len__(self):
        return len(self.imgs)


def get_dataset(dataset, batch_size, x_transforms, y_transforms):
    train_loader, val_loader, test_loader = None, None, None
    if dataset == 'liver':  # E:\代码\new\u_net_liver-master\data\liver\val
        train_dataset = LiverDataset(r"train", transform=x_transforms, target_transform=y_transforms)
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataset = LiverDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_loader = DataLoader(val_dataset, batch_size=1)
        test_loader = val_loader
    if dataset == "esophagus":
        train_dataset = esophagusDataset(r"train", transform=x_transforms, target_transform=y_transforms)
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataset = esophagusDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_loader = DataLoader(val_dataset, batch_size=1)
        test_loader = val_loader
    if dataset == "dsb2018_cell":
        train_dataset = dsb2018CellDataset(r"train", transform=x_transforms, target_transform=y_transforms)
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataset = dsb2018CellDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_loader = DataLoader(val_dataset, batch_size=1)
        test_loader = val_loader
    if dataset == 'corneal':
        train_dataset = CornealDataset(r'train', transform=x_transforms, target_transform=y_transforms)
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataset = CornealDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_loader = DataLoader(val_dataset, batch_size=1)
        test_dataset = CornealDataset(r"test", transform=x_transforms, target_transform=y_transforms)
        test_loader = DataLoader(test_dataset, batch_size=1)
    if dataset == 'drive_eye':
        train_dataset = DriveEyeDataset(r'train', transform=x_transforms, target_transform=y_transforms)
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataset = DriveEyeDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_loader = DataLoader(val_dataset, batch_size=1)
        test_dataset = DriveEyeDataset(r"test", transform=x_transforms, target_transform=y_transforms)
        test_loader = DataLoader(test_dataset, batch_size=1)
    if dataset == 'ISBI_cell':
        train_dataset = IsbiCellDataset(r'train', transform=x_transforms, target_transform=y_transforms)
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataset = IsbiCellDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_loader = DataLoader(val_dataset, batch_size=1)
        test_dataset = IsbiCellDataset(r"test", transform=x_transforms, target_transform=y_transforms)
        test_loader = DataLoader(test_dataset, batch_size=1)
    if dataset == 'kaggle_lung':
        train_dataset = LungKaggleDataset(r'train', transform=x_transforms, target_transform=y_transforms)
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataset = LungKaggleDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_loader = DataLoader(val_dataset, batch_size=1)
        test_dataset = LungKaggleDataset(r"test", transform=x_transforms, target_transform=y_transforms)
        test_loader = DataLoader(test_dataset, batch_size=1)
    if dataset == 'our_large':
        train_dataset = OurLargeDataset(r'train', transform=x_transforms, target_transform=y_transforms)
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataset = OurLargeDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_loader = DataLoader(val_dataset, batch_size=1)
        test_dataset = OurLargeDataset(r"test", transform=x_transforms, target_transform=y_transforms)
        test_loader = DataLoader(test_dataset, batch_size=1)
    if dataset == 'our_min':
        train_dataset = OurMinDataset(r'train', transform=x_transforms, target_transform=y_transforms)
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataset = OurMinDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_loader = DataLoader(val_dataset, batch_size=1)
        test_dataset = OurMinDataset(r"test", transform=x_transforms, target_transform=y_transforms)
        test_loader = DataLoader(test_dataset, batch_size=1)
    if dataset == 'TN-SCUI':
        train_dataset = TNSCUIDataset(r'train', transform=x_transforms, target_transform=y_transforms)
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataset = TNSCUIDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_loader = DataLoader(val_dataset, batch_size=1)
        test_dataset = TNSCUIDataset(r"test", transform=x_transforms, target_transform=y_transforms)
        test_loader = DataLoader(test_dataset, batch_size=1)
    if dataset == 'IDRiD':
        train_dataset = IDRiDDataset(r'train', transform=x_transforms, target_transform=y_transforms)
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataset = IDRiDDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_loader = DataLoader(val_dataset, batch_size=1)
        test_dataset = IDRiDDataset(r"test", transform=x_transforms, target_transform=y_transforms)
        test_loader = DataLoader(test_dataset, batch_size=1)
    return train_loader, val_loader, test_loader