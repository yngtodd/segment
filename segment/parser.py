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
    parser.add_argument('-d','--data', metavar='DIR', default='/raid/ChestXRay14/images',
                        help='path to dataset')
    args = parser.parse_args()
    return args
