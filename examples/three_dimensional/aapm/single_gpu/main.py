import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from segment.data import IRCAD3D
from segment.data.utils import train_valid_split

from segment.ml.models.three_dimensional.micronet import MicroUnet3D
from segment.ml import AverageMeter
from segment.ml.logging import Logger
from segment.ml.functional import dice_coefficient

from parser import parse_args


downsample_img = nn.AvgPool3d(2)
downsample_mask = nn.MaxPool3d(2)


def train(args, model, device, train_loader, optimizer, epoch, meters):
    trainloss = meters['loss']
    traindice = meters['dice']

    model.train()
    for batch_idx, (data, mask) in enumerate(train_loader):
        data = data.unsqueeze(1).float()
        mask = mask.unsqueeze(1).float()
        data = downsample_img(data)
        mask = downsample_mask(mask)
        data, mask = data.to(device), mask.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.binary_cross_entropy_with_logits(output, mask)
        dice = dice_coefficient(output, mask)
        loss.backward()
        optimizer.step()
        trainloss.update(loss.item())
        traindice.update(dice)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Dice: {:.6f}'.format(
                  epoch, batch_idx * len(data), len(train_loader.dataset),
                  100. * batch_idx / len(train_loader), loss.item(), traindice.avg))

            info = { 'train_loss': loss.item(), 'train_dice': traindice.avg }

            for tag, value in info.items():
                logger.scalar_summary(tag, value, epoch)

            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
                logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), epoch)

            imgs = output.squeeze(1)
            imgs = output.view(-1, 512, 512)[:2].detach().cpu().numpy()
            info = { 'segmentations': imgs }

            for tag, images in info.items():
                logger.image_summary(tag, images, epoch)


def test(args, model, device, test_loader, meters, epoch):
    testloss = meters['loss']
    testdice = meters['dice']

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (data, mask) in enumerate(test_loader):
            data = data.unsqueeze(1).float()
            mask = mask.unsqueeze(1).float()
            data = downsample_img(data)
            mask = downsample_mask(mask)
            data, mask = data.to(device), mask.to(device)
            output = model(data)
            loss = F.binary_cross_entropy_with_logits(output, mask, reduction='sum').item()
            test_loss += loss
            dice = dice_coefficient(output, mask)
            testdice.update(dice)
            testloss.update(loss)

            info = { 'test_loss': loss, 'test_dice': testdice.avg }

            for tag, value in info.items():
                logger.scalar_summary(tag, value, epoch)

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Average Dice Coefficient: {:.6f}\n'.format(
          testloss.avg, testdice.avg))


def main():
    args = parse_args()

    global logger
    logger = Logger(args.logpath)

    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = MicroUnet3D(n_channels=1, n_classes=1).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    dataset = IRCAD3D(args.datapath, tissue='bone')
    print(f'Segmenting {dataset.tissue}')

    trainset, testset = train_valid_split(dataset)
    trainloader = DataLoader(trainset, batch_size=args.batch_size)
    testloader = DataLoader(testset, batch_size=args.batch_size)

    train_meters = {
      'loss': AverageMeter('trainloss', args.meterpath),
      'dice': AverageMeter('traindice', args.meterpath)
    }

    test_meters = {
      'loss': AverageMeter('testloss', args.meterpath),
      'dice': AverageMeter('testdice', args.meterpath)
    }

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, trainloader, optimizer, epoch, train_meters)
        test(args, model, device, testloader, test_meters, epoch)

    train_meters['loss'].save()
    train_meters['dice'].save()
    test_meters['loss'].save()
    test_meters['dice'].save()


if __name__=='__main__':
    main()
