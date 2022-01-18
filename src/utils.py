"""
This Python script will be used for any logic that does not belong to
a distinct component of the learning process
"""

from pytorch_lightning.utilities.memory import garbage_collection_cuda
import time
import gc
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import rank_zero_only

def collect_garbage():
    garbage_collection_cuda()
    time.sleep(5)
    torch.cuda.empty_cache()
    garbage_collection_cuda()
    gc.collect()


# TODO put this in a Logger class
def log_results(logger, results, dm):

    # log training metrics
    results = results[0]
    for key in results.keys():
        res_dict = {'active_'+key: results[key], 'labelled_examples': len(dm.train.L)}
        logger.log(res_dict)

    return None


def log_percentages(indices, logger, dm, epoch):

    # TODO the condition below shouldn't be possible to begin with
    if len(indices) > len(dm.train.U):
        indices = indices[:len(dm.train.U)]

    new_examples = dm.train.U.iloc[indices]  # select queried examples from unlabelled pool
    percentages = new_examples['Dataset'].value_counts(normalize=True).to_dict()  # compute each dataset's share
    percentages['labelled_examples'] = len(dm.train.L) + len(new_examples)  # variable for x-axis: old L + new batch
    percentages['AL_iter'] = epoch
    logger.log(percentages)

    # TODO come up with a fancier way to plot, later.
    # labels = new_examples['Dataset'].unique()
    # values = new_examples['Dataset'].value_counts(normalize=True).to_list()
    # data = [[label, val] for (label, val) in zip(labels, values)]
    # table = logger.Table(data=data, columns=["label", "value"])
    # logger.log({"my_bar_chart_id": logger.plot.bar(table, "label", "value", title="Custom Bar Chart")})

    return None


@rank_zero_only
def c_print(*args, **kwargs):

    return print(*args, **kwargs)
