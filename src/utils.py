"""
This Python script will be used for any logic that does not belong to
a distinct component of the learning process
"""

def generate_run_id(args):

    id = 'seed_{}_'

# TODO put this in a Logger class
def log_results(logger, results, dm):

    results = results[0]
    for key in results.keys():
        res_dict = {'active_'+key: results[key], 'examples': len(dm.train.L)}
        logger.log(res_dict)

    return None
