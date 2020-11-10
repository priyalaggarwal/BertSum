import logging
import math
import os
import torch

logger = logging.getLogger()


def init_logger(log_file=None, log_file_level=logging.NOTSET):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger


def get_perplexity(loss, n_words):
    """ compute perplexity """
    return math.exp(min(loss / n_words, 100))


def save_checkpoint(model, step):
    model_state_dict = model.state_dict()

    checkpoint_dir = './checkpoints/'
    checkpoint_path = os.path.join(checkpoint_dir, 'model_step_%d.pt' % step)
    logger.info("Saving checkpoint %s" % checkpoint_path)

    if not os.path.exists(checkpoint_path):
        torch.save(model_state_dict, checkpoint_path)