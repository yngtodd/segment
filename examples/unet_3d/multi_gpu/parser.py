import argparse


def parse_args():
    """
    Parse Arguments for segmentation with UNet.

    Returns:
    -------
    * `args`: [argparse object]
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Segmentation parameters.')
    parser.add_argument('-d','--datapath', metavar='DIR', default='/raid/scratch/hinklejd/3D-IRCADb1',
                        help='path to dataset')
    parser.add_argument('-mtr','--meterpath', default='/home/ygx/experiments/segment',
                        help='path to save meter information')
    parser.add_argument('-log','--logpath', default='/home/ygx/segment/segment/learning/logging/logs',
                        help='path to save meter information')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--start_gpu', type=int, default=0, metavar='S',
                         help='First GPU to use. (default: 0)')
    parser.add_argument('--end_gpu', type=int, default=1, metavar='E',
                         help='Final GPU to use. (default: 1)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    return args
