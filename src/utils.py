"""
This Python script will be used for any logic that does not belong to
a distinct component of the learning process
"""

from pytorch_lightning.utilities.memory import garbage_collection_cuda
import time
import gc
import torch

def collect_garbage():
    garbage_collection_cuda()
    time.sleep(5)
    torch.cuda.empty_cache()
    garbage_collection_cuda()
    gc.collect()

def generate_run_id(args):

    id = 'seed_{}_'

# TODO put this in a Logger class
def log_results(logger, results, dm):

    results = results[0]
    for key in results.keys():
        res_dict = {'active_'+key: results[key], 'examples': len(dm.train.L)}
        logger.log(res_dict)

    return None
