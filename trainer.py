import os
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from loss import dice_coeff


class Trainer(object):

    def __init__(self, model, criterion, optimizer, scheduler, train_loader, val_loader, test_loader, args,
                 device, logging):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.args = args
        self.device = device
        self.logging = logging

        self.writer = SummaryWriter(log_dir=args.tensorboard_dir)

        inputs = next(iter(train_loader))[0]
        self.writer.add_graph(self.model, inputs.to(device, dtype=torch.float32))

        if args.weights:
            self.model.load_state_dict(torch.load(args.weights))
            logging.info(f'load weights:{args.weights} finish!')

        logging.info(f'''Starting training:
                    Model name:      {args.model}
                    Epochs:          {args.epoch}
                    Batch size:      {args.batch_size}
                    Learning rate:   {args.lr}
                    Dataset:         {args.dataset}
                    Checkpoints:     {args.ckp}
                    Device:          {device.type}
                    Input shape:     {list(inputs.shape[1:])}
                ''')

    def validate(self):
        self.model.eval()
        mask_type = torch.float32 if self.model.out_channels == 1 else torch.long
        n_val = len(self.val_loader)
        score = 0
        with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
            for batch in self.val_loader:
                images, masks = batch[0], batch[1]
                images = images.to(device=self.device, dtype=torch.float32)
                masks = masks.to(device=self.device, dtype=mask_type)

                with torch.no_grad():
                    pred = self.model(images)
                score += dice_coeff(pred, masks).item()
                pbar.update()
        self.model.train()
        return score / n_val

    def train(self):
        epochs = self.args.epoch
        threshold = self.args.threshold
        n_train = len(self.train_loader.dataset)
        step = 0
        best_score = 0
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            save = False
            # training
            with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
                for batch in self.train_loader:
                    images = batch[0]
                    masks = batch[1]
                    assert images.shape[1] == self.model.in_channels, \
                        f'modelwork has been defined with {self.model.in_channels} input channels, ' \
                        f'but loaded images have {images.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'

                    images = images.to(device=self.device, dtype=torch.float32)
                    # mask_type = torch.float32 if self.model.out_channels == 1 else torch.long
                    mask_type = torch.float32
                    masks = masks.to(device=self.device, dtype=mask_type)

                    self.optimizer.zero_grad()
                    masks_pred = self.model(images)
                    if self.args.deepsupervision:
                        masks_pred = masks_pred[-1]
                    loss = self.criterion(masks_pred, masks)

                    self.writer.add_scalar('Loss/train', loss.item(), step)
                    pbar.set_postfix(**{'loss(batch)': loss.item()})

                    if threshold:
                        if loss > threshold:
                            loss.backward()
                            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                            self.optimizer.step()
                            epoch_loss += loss.item()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                        self.optimizer.step()
                        epoch_loss += loss.item()
                    pbar.update(images.shape[0])
                    step = step + 1
            # eval
            if self.args.eval:
                for tag, value in self.model.named_parameters():
                    tag = tag.replace('.', '/')
                    self.writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), step)
                    self.writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), step)
                score = self.validate()
                if score > best_score:
                    best_score = score
                    save = True

                self.logging.info('Validation Dice Coeff: {}'.format(score))
                self.writer.add_scalar('Dice/val', score, step)
                self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], step)
                self.writer.add_images('images', images, step)
                # self.writer.add_images('masks/true', masks, step)
                self.writer.add_images('masks_MA/true', masks[:,0:1], step)
                self.writer.add_images('masks_HE/true', masks[:,1:2], step)
                self.writer.add_images('masks_EX/true', masks[:,2:3], step)
                self.writer.add_images('masks_SE/true', masks[:,3:4], step)
                # self.writer.add_images('masks/pred', masks_pred, step)
                self.writer.add_images('masks_MA/pred', masks_pred[:,0:1], step)
                self.writer.add_images('masks_HE/pred', masks_pred[:,1:2], step)
                self.writer.add_images('masks_EX/pred', masks_pred[:,2:3], step)
                self.writer.add_images('masks_SE/pred', masks_pred[:,3:4], step)
                self.scheduler.step(score)
            else:
                save = True
            # save checkpoint
            if self.args.ckp and save:
                try:
                    os.mkdir(self.args.ckp)
                except OSError:
                    pass
                model_name = f'{self.args.model}_{self.args.dataset}_'
                for folder, _, filenames in os.walk(self.args.ckp):
                    for filename in filenames:
                        if filename.startswith(model_name):
                            os.remove(os.path.join(folder, filename))
                torch.save(self.model.state_dict(), f'{self.args.ckp}/{model_name}{epoch + 1}_{int(time.time())}.pth')
                self.logging.info(f'Checkpoint {epoch + 1} saved !')
        self.writer.close()

    def test(self):

        pass
