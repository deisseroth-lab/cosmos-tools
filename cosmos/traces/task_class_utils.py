import numpy as np

def cross_val_mean_ev(traces, nfolds=5):
    """
    For each fold, compute the mean across
    training trials for each source.
    Then compute the explained variance of each test trial
    using the mean as the prediction.
    Also compute the R^2.

    Args:
        traces: [nsources x nframes x ntrials]
        nfold: number of cross-validation folds

    Returns:
        ev: [nsources x nfolds]
    """

    ntrials = traces.shape[2]
    nsources = traces.shape[0]
    shuff_trials = np.random.permutation(np.arange(ntrials))
    shuff_traces = traces[:, :, shuff_trials]

    ev_folds = np.zeros((nsources, nfolds))
    rsquared_folds = np.zeros((nsources, nfolds))
    ntest = int(ntrials / nfolds)
    for fold in range(nfolds):
        rolled_inds = np.roll(np.arange(ntrials), -ntest * fold)
        test_trials = rolled_inds[:ntest]
        train_trials = rolled_inds[ntest:]

        print(test_trials)
        #         print(train_trials)

        mean_train = np.mean(shuff_traces[:, :, train_trials], axis=2)[:, :,
                     np.newaxis]
        ytrue = shuff_traces[:, :, test_trials]
        ypred = np.tile(mean_train, (1, 1, len(test_trials)))

        for source in range(nsources):
            ntest = len(test_trials)
            trials_ev = np.zeros((ntest, 1))
            trials_rsquared = np.zeros((ntest, 1))
            for trial in range(ntest):
                trial_ytrue = ytrue[source, :, trial].flatten()
                trial_ypred = ypred[source, :, trial].flatten()
                if np.var(trial_ytrue) == 0:
                    ev = 0
                    rsquared = 0
                else:
                    rsquared = np.corrcoef(trial_ypred, trial_ytrue)[0, 1] ** 2
                    #                     r, p = scipy.stats.pearsonr(trial_ypred, trial_ytrue)
                    #                     rsquared = r**2
                    ev = 1 - np.var(trial_ytrue - trial_ypred) / np.var(
                        trial_ytrue)
                    # See https://www.biorxiv.org/content/biorxiv/early/2018/04/22/306019.full.pdf

                trials_ev[trial] = ev
                trials_rsquared[trial] = rsquared

            ev_folds[source, fold] = np.mean(trials_ev)
            rsquared_folds[source, fold] = np.mean(trials_rsquared)

    return ev_folds, rsquared_folds

def get_shuffle_for_parallel(input):
    """Wrapper for get_shuffle so that there is
    only a single input argument.

    :param input: (rates, rseed)
    :return: output of get_shuffle
    """
    rates = input[0]
    rseed = input[1]

    return get_shuffle(rates, rseed)


def get_shuffle(rates, rseed, ntrials_per_shuffle=52):
    """
    Compute explained variance and rsquared across sources
    and folds for a bootstrap shuffle.

    The shuffle works by, for each trial, rolling all
    sources along the time axis by a random shift.
    i.e. each trial is not randomly offset from the others
    and if they were previously aligned across trials
    they no longer should be.

    :param rates: [ncells x nframes x total_ntrials]
    :param rseed: int. which random seed.
    :param ntrials_per_shuffle: How many trials to include in one shuffle.
    :return:
    """

    np.random.seed(rseed)
    ntrials_total = rates.shape[2]

    trial_set = np.random.permutation(np.arange(ntrials_total))[:ntrials_per_shuffle]
    which_frames = np.arange(65, 200) # Only include odor period

    traces = np.zeros((rates.shape[0], len(which_frames), len(trial_set)))
    for trial in range(traces.shape[2]):
        shift = np.random.randint(0, len(which_frames)/2) ### Randomly roll each trial by at least two time frames.
        traces[:, :, trial] = np.roll(rates[:, which_frames, :][:, :, trial_set[trial]], shift, axis=1)

    ev_boot_shuff, rsquared_boot_shuff = cross_val_mean_ev(traces, nfolds=5)
    return [ev_boot_shuff, rsquared_boot_shuff]