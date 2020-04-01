"""
Helper functions for loading data saved out by bpod.
"""

import os, os.path
import scipy.io as spio
import numpy as np


def event_rate(event_times, bin_size, max_time):
    """
    Compute rate from a vector of times.
    Note: rate[0] is the rate of licks in the bin from time=0 to time=binSize

    :param event_times: ndarray of event times in seconds
    :param binSize: in seconds
    :param maxTime: in seconds
    :return rate: ndarray (of len max_time/bin_size) containing event rates
            t: time in seconds at the beginning of each time bin
    """
    rate = np.zeros(int(np.ceil((max_time - bin_size) / bin_size)))
    k = 0
    t = np.arange(0, max_time - bin_size, bin_size)
    for j in t:
        curr_win = [j, j + bin_size]
        rate[k] = np.sum(np.logical_and(event_times >= curr_win[0], event_times < curr_win[1])) / bin_size
        k = k + 1

    return rate, t


def get_state_times(allS, state_name, ind=0):
    """
    Extract time of entering (or exiting) a state for each trial.
    Wrapper function around _get_state_times.
    :param allS: either a single bpod data dict or array of dicts (loaded using load_bpod_data).
    :param state_name: name of the state you are querying i.e. 'Reward', 'NoReward'
    :param ind: if want the end of the state instead of the beginning, use ind=1
    :return: state_times: an array with the time for each trial. Entry is nan if did not enter state on trial.
    """

    if isinstance(allS, dict):
        return _get_state_times(allS, state_name, ind=ind)
    else:
        state_times = np.array([])
        for i in range(len(allS)):
            out = _get_state_times(allS[i], state_name, ind=ind)
            state_times = np.concatenate((state_times, out)) if state_times.size else out

        return state_times



def _get_state_times(S, state_name, ind=0):
    """
    Extract time of entering (or exiting) a state for each trial.
    :param S: bpod data (directly as loaded using load_bpod_data)
    :param state_name: name of the state you are querying i.e. 'Reward', 'NoReward'
    :param ind: if want the end of the state instead of the beginning, use ind=1
    :return: state_times: an array with the time for each trial. Entry is nan if did not enter state on trial.
    """
    ntrials = len(S['RawEvents']['Trial'])
    state_times = np.zeros(ntrials)
    for trial in range(ntrials):
        states = S['RawEvents']['Trial'][trial].States
        if state_name in states._fieldnames:
            state_time = getattr(states, state_name)
            state_times[trial] = state_time[ind]
        else:
            state_times[trial] = 0

    return state_times

def list_states(S):
    """
    Display a list of all possible behavioral states.
    :param S: bpod data (as loaded using load_bpod_data)
    """
    return S['RawEvents']['Trial'][0].States._fieldnames
    
def get_lick_rates(allS, bin_size=1.0 / 15, max_time=10, port='Port1In'):
    """
    Return rate of licks, directly from the bpod data file (concatenates multiple sessions if necessary).
    Note: rate[0] is the rate of licks in the bin from time=0 to time=binSize

    Wrapper function to concatenate together _get_lick_rates for multiple sessions (where the multiple sessions
    are exactly the same and taken on the same day, but were split up by bpod).
    :param allS: either a single bpod data dict or array of dicts (loaded using load_bpod_data).
    :param binSize: in seconds
    :param maxTime: in seconds
    :param port: which bpod port was recording licks
    :return all_lick_rates: a matrix [trial x time] containing lick rates
            t: the time associated with each bin in all_lick_rates
            all_lick_times: a list containing, for each session, a dict containing actual lick times for each trial
    """
    print('lick rates bin size: ' +str(bin_size))
    if isinstance(allS, dict):
        return _get_lick_rates(allS, bin_size=bin_size, max_time=max_time, port=port)
    else:
        all_lick_rates = np.array([])
        t = np.array([])
        all_lick_times = []
        for i in range(len(allS)):
            out = _get_lick_rates(S=allS[i], bin_size=bin_size, max_time=max_time, port=port)
            all_lick_rates = np.concatenate((all_lick_rates, out[0])) if all_lick_rates.size else out[0]
            t = out[1]
            if not all_lick_times:
                all_lick_times = [out[2]]
            else:
                all_lick_times.append(out[2])

        return all_lick_rates, t, all_lick_times


def _get_lick_rates(S, bin_size=1.0 / 15, max_time=10, port='Port1In'):
    """
    Return rate of licks, directly from the bpod data file.
    Note: rate[0] is the rate of licks in the bin from time=0 to time=binSize

    :param S: bpod data (directly as loaded using load_bpod_data)
    :param binSize: in seconds
    :param maxTime: in seconds
    :param port: which bpod port was recording licks
    :return all_lick_rates: a matrix [trial x time] containing lick rates
            t: the time associated with each bin in all_lick_rates
            all_lick_times: a dict containing actual lick times for each trial
    """

    all_lick_times = {}
    ntrials = len(S['RawEvents']['Trial'])
    all_lick_rates = np.zeros((ntrials, int(np.ceil((max_time - bin_size) / bin_size))))
    for trial in range(ntrials):
        events = S['RawEvents']['Trial'][trial].Events
        if port in events._fieldnames:
            lick_times = getattr(events, port)
            all_lick_times[trial] = lick_times
            lick_rate, t = event_rate(lick_times, bin_size, max_time)
            all_lick_rates[trial][:] = lick_rate
        else:
            t = np.arange(0, max_time - bin_size, bin_size)

    return all_lick_rates, t, all_lick_times

def lick_times_to_matrix(lick_times, max_t=7, ntrials=None, hz=1000):
    """
    Converts an array with lick times in each trial
    into a discretized matrix (1ms per bin), where 
    a 1 indicates that lick occurred during that time bin.
    :param lick_times: a dict with an key representing each trial 
                        that contains an array with the lick times
                        of that trial.
    :param max_t: time, in seconds, for samples. 
    :param ntrials: number of rows of the matrix
    :param hz: number of bins per second 
    
    :returns lick_mat: [ntrials x max_t*hz] binary matrix
    :returns lick_mat_t: timestamp in s for each column of matrix
    """
    
    if ntrials is None:
        ntrials = np.amax(np.array(list(lick_times.keys())))
    
    lick_mat = np.zeros((ntrials, max_t*hz))
    for trial in lick_times.keys():
        if trial < ntrials:
            times = lick_times[trial]
            inds = np.round(times*hz).astype(int)
            if np.isscalar(inds):
                if inds < np.shape(lick_mat)[1]:
                    lick_mat[trial, inds] = 1
            else:
                for ind in inds:
                    if ind < np.shape(lick_mat)[1]:
                        lick_mat[trial, ind] = 1
    lick_mat_t = np.linspace(0, max_t, max_t*hz)
    return lick_mat, lick_mat_t
    

def make_bpod_data_path(base_path, mouse_name, protocol_name, date, session):
    """
    Returns path to bpod mat file containing all behavior data.
    """
    fname = mouse_name+'_'+protocol_name+'_'+date+'_Session'+str(session)+'.mat'
    return os.path.join(base_path, mouse_name, protocol_name,
                         'Session Data', fname)

def make_bpod_img_paths(base_path, mouse_name, protocol_name, date, session):
    """
    Returns a list with paths to the each directory containing an image stack
    corresponding to the behavior trials in the specified session.
    """
    session_path = os.path.join(base_path, mouse_name, protocol_name,
                                date, 'Session'+str(session))
    trials = os.listdir(session_path)

    img_paths = []
    for t in trials:
        dname = mouse_name+'_'+protocol_name+'_'+date+'_Session'+str(session)\
                +'-'+t[-3:]
        img_paths.append(os.path.join(session_path, t, dname))

    return img_paths

def make_bpod_processed_img_paths(img_path, processed_img_path, bpod_img_fullpaths):
    """
    Returns a list with paths to the directory where processed (i.e. motion
    corrected) videos corresponding to each behavior trial should will be saved.
    """
    processed_paths = []

    for p in bpod_img_fullpaths:
        processed_paths.append(p.replace(img_path,processed_img_path))
        if not os.path.exists(processed_paths[-1]):
            os.makedirs(processed_paths[-1])

    return processed_paths

def load_bpod_data(bpod_data_fullpath):
    """
    Loads bpod mat file into a dict.
    Note: the 'TrialSettings' entry of the mat does not seem to load in an
    accessible manner, but everything else seems good.

    :param bpod_data_fullpath: path to the .mat file
    :return: a dict containing contents of the mat file.
    """
    bpod_data = loadmat(bpod_data_fullpath)
    return bpod_data['SessionData']


def get_full_bpod_path(base_path, bpod_fname):
    """
    Get full path to bpod file based just on
    the name of the file and the base path
    to the folder containing all bpod data.
    :param base_path: i.e. '/Dropbox/Bpod_data/'
    :param bpod_fname:  i.e. 'm10_protocol_20190101_00000.mat'
    :return:
    """
    bpod_s = bpod_fname.split('_')
    mouse_name = bpod_s[0]
    date = bpod_s[-2]
    protocol_name = '_'.join(bpod_s[1:-2])

    fpath = os.path.join(base_path, mouse_name, protocol_name, 'Session Data',
                         bpod_fname + '.mat')
    return fpath


### Helper functions ###

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    #data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict