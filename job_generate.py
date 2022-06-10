from itertools import combinations_with_replacement, permutations
import argparse


def generate_perms(config):
    # initializing N
    N = 11

    # Initialize K
    K = len(config.train_sets)

    res = list(combinations_with_replacement(range(N), K))
    frac_combs = []
    for comb in res:
        new_comb = []
        for val in comb:
            new_comb.append(val)
        frac_combs.append(new_comb)

    valid = []
    for item in frac_combs:
        if sum(item) == 4:
            perm = list(permutations(item))
            seen = []
            for comb in perm:
                if comb not in seen:
                    seen.append(comb)
            valid.extend(seen)

    valid_new = []
    for comb in valid:
        # print(comb)
        new_comb = []
        for i in range(len(comb)):
            val = comb[i]
            data_id = config.train_sets[i]
            new_comb.append(data_id + '_' + str(config.increments * val))
        # print(new_comb)
        valid_new.append(new_comb)

    return valid_new

def main(config):

    job = ''
    unique_perms = generate_perms(config)

    with open("new_jobs/hparams/unique_perm_hyperparams.txt", "w") as text_file:
        for seed in config.seeds:
            for train_sets in unique_perms:

                dev_sets = []
                for data_fraction_pair in train_sets:

                    dataset = data_fraction_pair.split('_')[0]
                    fraction = data_fraction_pair.split('_')[1]

                    if config.only_dev_on_train:
                        if float(fraction) > 0:
                            dev_sets.append(dataset)

                    else:
                        dev_sets.append(dataset)


                train_sets = ','.join(train_sets)
                dev_sets = ','.join(dev_sets)
                text_file.write('--train_sets={} --dev_sets={} --test_sets={} --seed={}\n'.format(train_sets,
                                                                                                  dev_sets,
                                                                                                  config.test_sets,
                                                                                                  seed))
        text_file.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--train_sets', default='SNLI,ANLI,WANLI', type=str)
    parser.add_argument('--dev_sets', default='SNLI,ANLI,WANLI', type=str)
    parser.add_argument('--test_sets', default='SNLI,ANLI,MNLI,WANLI')
    parser.add_argument('--increments', default=0.25, type=float)
    parser.add_argument('--seeds', default='42,43,44,45,46', type=str)
    parser.add_argument('--only_dev_on_train', default=False, type=bool)

    config = parser.parse_args()

    config.train_sets = config.train_sets.split(',')
    config.seeds = [int(val) for val in config.seeds.split(',')]

    main(config)
