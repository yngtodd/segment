import time
import argparse


def parse_args():
    """
    Parse Arguments for segmentation with UNet.

    Returns:
    -------
    * `args`: [argparse object]
        Parsed arguments.
    """
    strtime = time.strftime("%Y-%m-%d")
    parser = argparse.ArgumentParser(description='Segmentation parameters.')
    parser.add_argument('-d','--datapath', metavar='DIR', default='/raid/scratch/hinklejd/3D-IRCADb1',
                        help='path to dataset')
    parser.add_argument('-mtr','--meterpath', default='/home/ygx/experiments/segment',
                        help='path to save meter information')
    parser.add_argument('-log','--logpath', default='/home/ygx/experiments/segment/two_dimensional',
                        help='path to save meter information')
    parser.add_argument('--savepath', default='/home/ygx/segment/saves/two_dimensional/ircad',
                        help='path to save model/optimizer states')
    parser.add_argument('--savefile', default=f'ircad2d_{strtime}',
                        help='filename to save model/optimizer states')
    parser.add_argument('--checkpoint', action='store_true', default=False,
                        help='Whether to checkpoint model')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Whether to resume from previously saved model')
    parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    return args
