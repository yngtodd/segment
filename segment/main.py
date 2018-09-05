import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from segment.data import IRCAD2D
from segment.data.utils import train_valid_split

from segment.learning import UNet
from segment.learning import AverageMeter 
from segment.learning.functional import dice_coeff

from parser import parse_args


def train(args, model, device, train_loader, optimizer, epoch, meters):
    trainloss = meters['loss']
    traindice = meters['dice']

    model.train()
    for batch_idx, (data, mask) in enumerate(train_loader):
        data = data.unsqueeze(1).float()
        mask = mask.unsqueeze(1).float()
        data, mask = data.to(device), mask.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.binary_cross_entropy_with_logits(output, mask)
        dice = dice_coeff(output, mask)
        loss.backward()
        optimizer.step()
        trainloss.update(loss.item())
        dice = dice.detach()
        traindice.update(dice)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Dice: {:.6f}'.format(
                  epoch, batch_idx * len(data), len(train_loader.dataset),
                  100. * batch_idx / len(train_loader), loss.item(), traindice.avg))


def test(args, model, device, test_loader, meters):
    testdice = meters['dice']

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for idx, (data, mask) in enumerate(test_loader):
            data = data.unsqueeze(1).float()
            mask = mask.unsqueeze(1).float()
            data, mask = data.to(device), mask.to(device)
            output = model(data)
            test_loss += F.binary_cross_entropy_with_logits(output, mask, reduction='sum').item()
            dice = dice_coeff(output, mask)
            testdice.update(dice)

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Average Dice Coefficient: {:.6f})\n'.format(
          test_loss, traindice.avg))


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = UNet(n_channels=1, n_classes=1).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    dataset = IRCAD2D(args.datapath, 'bone')
    trainset, testset = train_valid_split(dataset)
    
    trainloader = DataLoader(trainset)
    testloader = DataLoader(testset)

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
        test(args, model, device, testloader, test_meters)

    train_meters['loss'].save()
    train_meters['dice'].save()
    test_meters['dice'].save()


if __name__=='__main__':
    main()
