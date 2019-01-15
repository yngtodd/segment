import logging
import logging.config


logging.config.fileConfig('logging.conf', defaults={'logfilename': './logs/main.log'})
logger = logging.getLogger(__name__)


class record:
    """
    Decorator to log train/val runtime and results.
    """
    def __init__(self, function):
        self.function = function

    def __call__(self, *args, **kwargs):
        logging.basicConfig(filename=f'{self.function.__name__}.log', level=logging.INFO)

        epoch = kwargs['epoch']

        t1 = time.time()
        loss = self.function(*args, *kwargs)
        t2 = time.time() - t1

        logging.info(
          f'{self.function.__name__} '\
          f'epoch: {epoch} runtime: {t2:.4f} seconds '\
          f'loss: {loss}'
        )

        return loss
