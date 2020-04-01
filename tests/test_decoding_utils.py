import cosmos.traces.decoding_utils as utils
import numpy as np
from cosmos.traces.cosmos_traces import CosmosTraces
import os


def test_split_dataset_1():
    """
    Ensure that split_dataset properly returns datasets that are
    shifted temporally in the correct direction.
    2 categories (lick vs no-lick).
    """
    # Load test traces dataset.
    dataset = {'date': '20180227', 'name': 'testdata_cux2m72_COSMOSTrainMultiBlockGNG_1'}
    filename = __file__
    test_dir, _ = os.path.split(filename)
    dataset['data_root'] = os.path.join(test_dir, 'test_cosmos_dataset')
    dataset['fig_save_dir'] = test_dir
    # dataset['behavior_dir'] = None # Have not yet made a test bpod dataset.

    CT = CosmosTraces(dataset, nt_per_trial=4, ntrials=2)
    X = CT.C.T
    Y = np.zeros((X.shape[0], 2))
    Y[:, 0] = 1

    # Setup artificial lick covariate that you intent to predict.
    lick_inds = [100, 200, 220, 224, 225, 270, 275, 300, 450, 451, 460]
    Y[lick_inds, 0] = 0
    Y[lick_inds, 1] = 1

    # Set options for splitting dataset into train/test/validate.
    splitopts = utils.default_splitopts
    splitopts['bins_before'] = 3
    splitopts['bins_current'] = 0
    splitopts['bins_after'] = 0

    splitopts['train_inds'] = np.arange(10, 250)
    splitopts['test_inds'] = np.arange(250, 370)
    splitopts['valid_inds'] = np.arange(370, 490)
    splitopts['standardize_X'] = False
    splitopts['center_Y'] = False

    neural_data = X
    Y_full = Y
    data_split = utils.split_dataset(neural_data, Y_full, splitopts, do_debug=False)

    assert(data_split['X_train'].shape == (240, 3, 8))
    assert(data_split['X_test'].shape == (120, 3, 8))
    assert(data_split['X_valid'].shape == (120, 3, 8))
    assert(data_split['Y_train'].shape == (240, 2))
    assert(data_split['Y_test'].shape == (120, 2))
    assert(data_split['Y_valid'].shape == (120, 2))

    Y_train = data_split['Y_train']
    Y_test = data_split['Y_test']
    Y_valid = data_split['Y_valid']
    train_inds = splitopts['train_inds']
    test_inds = splitopts['test_inds']
    valid_inds = splitopts['valid_inds']
    bins_before = splitopts['bins_before']
    assert(Y_train[np.where(train_inds == 100)[0]- bins_before, 1] == 1)
    assert(Y_test[np.where(test_inds == 300)[0]- bins_before, 1] == 1)
    assert(Y_valid[np.where(valid_inds == 460)[0]- bins_before, 1] == 1)
    # These above asserts are because split_dataset cuts off bins_before indices
    # from the beginning of the array.

    # Now test whether X_train matches up with the correct lick features.
    X_train = data_split['X_train']
    X_test = data_split['X_test']
    X_valid = data_split['X_valid']

    assert(Y[100, 1] ==
           Y_train[np.where(train_inds == 100)[0] - bins_before, 1])
    assert(X[97, 0] ==
           X_train[np.where(train_inds == 100)[0]- bins_before, 0, 0])
    assert(X[98, 0] ==
           X_train[np.where(train_inds == 100)[0]- bins_before, 1, 0])
    assert(X[99, 0] ==
           X_train[np.where(train_inds == 100)[0]- bins_before, 2, 0])

    assert(Y[275, 1] ==
           Y_test[np.where(test_inds == 275)[0] - bins_before, 1])
    assert(X[272, 0] ==
           X_test[np.where(test_inds == 275)[0]- bins_before, 0, 0])
    assert(X[273, 0] ==
           X_test[np.where(test_inds == 275)[0]- bins_before, 1, 0])
    assert(X[274, 0] ==
           X_test[np.where(test_inds == 275)[0]- bins_before, 2, 0])

    assert(Y[460, 1] ==
           Y_valid[np.where(valid_inds == 460)[0] - bins_before, 1])
    assert(X[457, 0] ==
           X_valid[np.where(valid_inds == 460)[0]- bins_before, 0, 0])
    assert(X[458, 0] ==
           X_valid[np.where(valid_inds == 460)[0]- bins_before, 1, 0])
    assert(X[459, 0] ==
           X_valid[np.where(valid_inds == 460)[0]- bins_before, 2, 0])

def test_split_dataset_2():
    """
    Additional testing that split_dataset properly returns the correct
    timing of shifted datapoints. 2 categories (lick vs no-lick).
    """
    dataset = {'date': '20180227',
               'name': 'testdata_cux2m72_COSMOSTrainMultiBlockGNG_1'}
    filename = __file__
    test_dir, _ = os.path.split(filename)
    dataset['data_root'] = os.path.join(test_dir, 'test_cosmos_dataset')
    dataset['fig_save_dir'] = test_dir
    # dataset['behavior_dir'] = None # Have not yet made a test bpod dataset.

    CT = CosmosTraces(dataset, nt_per_trial=4, ntrials=2)
    X = CT.C.T
    Y = np.zeros((X.shape[0], 2))
    Y[:, 0] = 1

    lick_inds = [100, 200, 220, 224, 225, 270, 275, 300, 450, 451, 460]
    Y[lick_inds, 0] = 0
    Y[lick_inds, 1] = 1

    splitopts = utils.default_splitopts
    splitopts['bins_before'] = 4
    splitopts['bins_current'] = 1 # This is changed from test_split_dataset_1().
    splitopts['bins_after'] = 0

    splitopts['train_inds'] = np.arange(10, 250)
    splitopts['test_inds'] = np.arange(250, 370)
    splitopts['valid_inds'] = np.arange(370, 490)
    splitopts['standardize_X'] = False
    splitopts['center_Y'] = False

    neural_data = X
    Y_full = Y

    data_split = utils.split_dataset(neural_data, Y_full, splitopts,
                                     do_debug=False)

    assert (data_split['X_train'].shape == (240, 5, 8))
    assert (data_split['X_test'].shape == (120, 5, 8))
    assert (data_split['X_valid'].shape == (120, 5, 8))
    assert (data_split['Y_train'].shape == (240, 2))
    assert (data_split['Y_test'].shape == (120, 2))
    assert (data_split['Y_valid'].shape == (120, 2))

    Y_train = data_split['Y_train']
    Y_test = data_split['Y_test']
    Y_valid = data_split['Y_valid']
    train_inds = splitopts['train_inds']
    test_inds = splitopts['test_inds']
    valid_inds = splitopts['valid_inds']
    bins_before = splitopts['bins_before']
    assert (Y_train[np.where(train_inds == 100)[0] - bins_before, 1] == 1)
    assert (Y_test[np.where(test_inds == 300)[0] - bins_before, 1] == 1)
    assert (Y_valid[np.where(valid_inds == 460)[0] - bins_before, 1] == 1)
    # These above asserts are because split_dataset cuts off bins_before indices
    # from the beginning of the array.

    # Now test whether X_train matches up with the correct lick features.
    X_train = data_split['X_train']
    X_test = data_split['X_test']
    X_valid = data_split['X_valid']

    assert (Y[100, 1] ==
            Y_train[np.where(train_inds == 100)[0] - bins_before, 1])
    assert (X[96, 0] ==
            X_train[np.where(train_inds == 100)[0] - bins_before, 0, 0])
    assert (X[97, 0] ==
            X_train[np.where(train_inds == 100)[0] - bins_before, 1, 0])
    assert (X[98, 0] ==
            X_train[np.where(train_inds == 100)[0] - bins_before, 2, 0])
    assert (X[99, 0] ==
            X_train[np.where(train_inds == 100)[0] - bins_before, 3, 0])
    assert (X[100, 0] ==
            X_train[np.where(train_inds == 100)[0] - bins_before, 4, 0])

    assert (Y[275, 1] ==
            Y_test[np.where(test_inds == 275)[0] - bins_before, 1])
    assert (X[271, 0] ==
            X_test[np.where(test_inds == 275)[0] - bins_before, 0, 0])
    assert (X[272, 0] ==
            X_test[np.where(test_inds == 275)[0] - bins_before, 1, 0])
    assert (X[273, 0] ==
            X_test[np.where(test_inds == 275)[0] - bins_before, 2, 0])
    assert (X[274, 0] ==
            X_test[np.where(test_inds == 275)[0] - bins_before, 3, 0])
    assert (X[275, 0] ==
            X_test[np.where(test_inds == 275)[0] - bins_before, 4, 0])

    assert (Y[460, 1] ==
            Y_valid[np.where(valid_inds == 460)[0] - bins_before, 1])
    assert (X[456, 0] ==
            X_valid[np.where(valid_inds == 460)[0] - bins_before, 0, 0])
    assert (X[457, 0] ==
            X_valid[np.where(valid_inds == 460)[0] - bins_before, 1, 0])
    assert (X[458, 0] ==
            X_valid[np.where(valid_inds == 460)[0] - bins_before, 2, 0])
    assert (X[459, 0] ==
            X_valid[np.where(valid_inds == 460)[0] - bins_before, 3, 0])
    assert (X[460, 0] ==
            X_valid[np.where(valid_inds == 460)[0] - bins_before, 4, 0])


def test_split_dataset_3():
    """
    Additional testing that split_dataset properly returns the correct
    timing of shifted datapoints. Test with no-lick and 3 spouts
    (four categories).
    """
    dataset = {'date': '20180227',
               'name': 'testdata_cux2m72_COSMOSTrainMultiBlockGNG_1'}
    filename = __file__
    test_dir, _ = os.path.split(filename)
    dataset['data_root'] = os.path.join(test_dir, 'test_cosmos_dataset')
    dataset['fig_save_dir'] = test_dir
    # dataset['behavior_dir'] = None # Have not yet made a test bpod dataset.

    CT = CosmosTraces(dataset, nt_per_trial=4, ntrials=2)
    X = CT.C.T
    Y = np.zeros((X.shape[0], 4))
    Y[:, 0] = 1

    spout_1_inds = [100, 200, 220, 224, 225, 270, 275, 300, 450, 451, 460]
    Y[spout_1_inds, 0] = 0
    Y[spout_1_inds, 1] = 1

    spout_2_inds = [150, 280, 420]
    Y[spout_2_inds, 0] = 0
    Y[spout_2_inds, 2] = 1

    spout_3_inds = [160, 291, 425]
    Y[spout_3_inds, 0] = 0
    Y[spout_3_inds, 3] = 1

    splitopts = utils.default_splitopts
    splitopts['bins_before'] = 2
    splitopts['bins_current'] = 1  # This is changed from test_split_dataset_1().
    splitopts['bins_after'] = 2

    splitopts['train_inds'] = np.arange(10, 250)
    splitopts['test_inds'] = np.arange(250, 370)
    splitopts['valid_inds'] = np.arange(370, 490)
    splitopts['standardize_X'] = False
    splitopts['center_Y'] = False

    neural_data = X
    Y_full = Y

    data_split = utils.split_dataset(neural_data, Y_full, splitopts,
                                     do_debug=False)

    assert (data_split['X_train'].shape == (240, 5, 8))
    assert (data_split['X_test'].shape == (120, 5, 8))
    assert (data_split['X_valid'].shape == (120, 5, 8))
    assert (data_split['Y_train'].shape == (240, 4))
    assert (data_split['Y_test'].shape == (120, 4))
    assert (data_split['Y_valid'].shape == (120, 4))

    Y_train = data_split['Y_train']
    Y_test = data_split['Y_test']
    Y_valid = data_split['Y_valid']
    train_inds = splitopts['train_inds']
    test_inds = splitopts['test_inds']
    valid_inds = splitopts['valid_inds']
    bins_before = splitopts['bins_before']
    assert (Y_train[np.where(train_inds == 100)[0] - bins_before, 1] == 1)
    assert (Y_test[np.where(test_inds == 300)[0] - bins_before, 1] == 1)
    assert (Y_valid[np.where(valid_inds == 460)[0] - bins_before, 1] == 1)
    assert (Y_train[np.where(train_inds == 150)[0] - bins_before, 2] == 1)
    assert (Y_test[np.where(test_inds == 280)[0] - bins_before, 2] == 1)
    assert (Y_valid[np.where(valid_inds == 420)[0] - bins_before, 2] == 1)
    assert (Y_train[np.where(train_inds == 160)[0] - bins_before, 3] == 1)
    assert (Y_test[np.where(test_inds == 291)[0] - bins_before, 3] == 1)
    assert (Y_valid[np.where(valid_inds == 425)[0] - bins_before, 3] == 1)
    # These above asserts are because split_dataset cuts off bins_before indices
    # from the beginning of the array.

    # Now test whether X_train matches up with the correct lick features.
    X_train = data_split['X_train']
    X_test = data_split['X_test']
    X_valid = data_split['X_valid']

    assert (Y[100, 1] ==
            Y_train[np.where(train_inds == 100)[0] - bins_before, 1])
    assert (X[98, 0] ==
            X_train[np.where(train_inds == 100)[0] - bins_before, 0, 0])
    assert (X[99, 0] ==
            X_train[np.where(train_inds == 100)[0] - bins_before, 1, 0])
    assert (X[100, 0] ==
            X_train[np.where(train_inds == 100)[0] - bins_before, 2, 0])
    assert (X[101, 0] ==
            X_train[np.where(train_inds == 100)[0] - bins_before, 3, 0])
    assert (X[102, 0] ==
            X_train[np.where(train_inds == 100)[0] - bins_before, 4, 0])

    assert (Y[280, 2] ==
            Y_test[np.where(test_inds == 280)[0] - bins_before, 2])
    assert (X[278, 0] ==
            X_test[np.where(test_inds == 280)[0] - bins_before, 0, 0])
    assert (X[279, 0] ==
            X_test[np.where(test_inds == 280)[0] - bins_before, 1, 0])
    assert (X[280, 0] ==
            X_test[np.where(test_inds == 280)[0] - bins_before, 2, 0])
    assert (X[281, 0] ==
            X_test[np.where(test_inds == 280)[0] - bins_before, 3, 0])
    assert (X[282, 0] ==
            X_test[np.where(test_inds == 280)[0] - bins_before, 4, 0])

    assert (Y[425, 3] ==
            Y_valid[np.where(valid_inds == 425)[0] - bins_before, 3])
    assert (X[423, 0] ==
            X_valid[np.where(valid_inds == 425)[0] - bins_before, 0, 0])
    assert (X[424, 0] ==
            X_valid[np.where(valid_inds == 425)[0] - bins_before, 1, 0])
    assert (X[425, 0] ==
            X_valid[np.where(valid_inds == 425)[0] - bins_before, 2, 0])
    assert (X[426, 0] ==
            X_valid[np.where(valid_inds == 425)[0] - bins_before, 3, 0])
    assert (X[427, 0] ==
            X_valid[np.where(valid_inds == 425)[0] - bins_before, 4, 0])


def test_get_data_for_decoding():
    dataset = {'date': '20180227', 'name': 'testdata_cux2m72_COSMOSTrainMultiBlockGNG_1'}
    filename = __file__
    test_dir, _ = os.path.split(filename)
    dataset['data_root'] = os.path.join(test_dir, 'test_cosmos_dataset')
    dataset['fig_save_dir'] = test_dir
    # dataset['behavior_dir'] = None

    CT = CosmosTraces(dataset, nt_per_trial=4, ntrials=2)
    class BpodDataset:
        def __init__(self):
            self.spout_lick_times = None

    BD = BpodDataset()
    BD.spout_lick_times = {0: {0: np.array([2.7, 2.85, 2.94]),
                               },
                           1: {},
                           2: {0: np.array([1.4, 2.4, 2.5]) },
                           3: {0: np.array([2.1]),
                               1: np.array([2.1, 2.4, 2.7, 3.01])}
                           }
    CT.bd = BD

    assert(CT.Ct.shape == (8, 117, 2))
    assert(CT.St.shape == (8, 117, 2))
    assert(CT.Ft.shape == (8, 117, 2))

    default_data_opts = {'decoding_set': 1,
                         # ID number of which data labeling to decode.
                         # See get_data_for_decoding for details.
                         'train_feat': 'spikes',
                         # 'spikes', 'smooth_spikes', or 'fluor'
                         'train_frac': 0.5,
                         'test_frac': 0.25,
                         'valid_frac': 0.25,
                         'remove_multi_licks': False,
                         'rand_seed': 0,
                         'bins_current': 0,  # include frame of each event
                         'bins_before': 2,  # frames before the event
                         'bins_after': 0,  # frames after the event
                         'standardize_X': True,  # u=0, s=1
                         'center_Y': True}  # u=0

    opts = default_data_opts
    opts['decoding_set'] = 1
    opts['train_frac'] = 2/3
    opts['test_frac'] = 1/3
    opts['valid_frac'] = 0
    opts['bins_current'] = 0  # include frame of each event
    opts['bins_before'] = 2  # frames before the event
    opts['bins_after'] = 0
    data_split = utils.get_data_for_decoding(CT, opts, do_debug=False)
    assert(data_split['X_test'].shape[1] == 2)
    assert(data_split['Y_test'].shape[1] == 4) # For decoding_set = 1


def test_evenly_space_trial_types():
    trial_types = np.array(
        [3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
         3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3,
         3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4])

    trials = np.arange(len(trial_types))

    reordered_trials = utils.evenly_space_trial_types(trials, trial_types,
                                                      do_random_permute=True)

    reordered_trial_types = trial_types[reordered_trials]
    assert(3 in reordered_trial_types[-6:])
    assert(1 in reordered_trial_types[-6:])
    assert(4 in reordered_trial_types[-6:])

    assert(3 in reordered_trial_types[:6])
    assert(1 in reordered_trial_types[:6])
    assert(4 in reordered_trial_types[:6])


def test_get_neuron_subsets_1():
    all_neurons = np.arange(1241)
    for subset_size in [1, 10, 11, 21, 801, 2000]:
        for nrepeats in  [1, 3, 5]:
            subsets = utils.get_neuron_subsets(all_neurons, subset_size, nrepeats)
            assert(len(subsets) == np.ceil(len(all_neurons)/subset_size)*nrepeats)

            ### Ensure all subsets are of the correct length.
            a = [len(x) for x in subsets]
            u = np.unique(np.array(a))
            assert(len(u) == 1)
            if subset_size <= len(all_neurons):
                assert (u[0] == subset_size)
            else:
                assert (u[0] == len(all_neurons))

            ### Ensure each neuron is in a subset at least nrepeats times.
            subsets_flat = np.concatenate(subsets)
            for neuron in all_neurons:
                assert(len(np.where(subsets_flat == neuron)[0]) >= nrepeats)

def test_get_neurons_for_decoding_1():
    # get_neurons_for_decoding(neuron_opts, CT, nrepeats)
    pass


def test_get_decoding_experiment_group_1():
    # experiment_group = get_decoding_experiment_group(expt_param, CT,
    #                                                  decoding_save_dir)
    # Test for coverage
    pass



def test_get_source_discriminativity_1():
    #     ### Manually check that the pvalue ordering actually makes sense.
    #     ### TODO: make into a unit test.
    #     i = p_ordering[-2]
    #     a =[0]*4
    #     a[0] = np.sum(X[np.where(y==0)[0], 2, i])/len(X[np.where(y==0)[0], 2, i])
    #     a[1] = np.sum(X[np.where(y==1)[0], 2, i])/len(X[np.where(y==1)[0], 2, i])
    #     a[2] = np.sum(X[np.where(y==2)[0], 2, i])/len(X[np.where(y==2)[0], 2, i])
    #     a[3] = np.sum(X[np.where(y==3)[0], 2, i])/len(X[np.where(y==3)[0], 2, i])
    #     print(a)
    pass