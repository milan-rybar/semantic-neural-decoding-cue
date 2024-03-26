import hashlib
import json
import logging
import os
import pickle

from scipy import signal
from scipy.stats import binom


def unique_dict_id(dict_value):
    # Hash randomisation is turned on by default in Python 3 => hashlib
    return hashlib.sha1(json.dumps(dict_value, sort_keys=True).encode('utf-8')).hexdigest()


def make_dirs(path):
    if not os.path.exists(path):
        try:  # in case that in parallel multiple workers try to do the dir
            os.makedirs(path)
            logging.debug('created %s', path)
        except:
            logging.warning('Attempt to create dir "%s" that already exists.', path)


def pickle_dict(file_path, data_dict):
    with open(file_path, 'wb') as f:
        pickle.dump(data_dict, f, -1)  # with pickle.HIGHEST_PROTOCOL


def unpickle_dict(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def high_pass(raw, high_pass_cut, high_pass_order):
    """
    Apply high-pass filter to EEG data.

    :param raw: raw data
    :type raw: `mne.io.base.BaseRaw`
    """
    fs = raw.info['sfreq']
    assert fs == 2048.0

    # high pass filter
    high_pass_sos = signal.butter(high_pass_order, high_pass_cut / (fs / 2),
                                  'high', output='sos')

    def filter_data(timeseries):
        return signal.sosfiltfilt(high_pass_sos, timeseries)

    raw.apply_function(filter_data, n_jobs=1)


def borderline_significance_level(alpha, n_trials):
    """
    Compute significance borderline for a random classifier with the desired level.

    :param alpha: significance level (e.g., 0.05, 0.01, ...)
    :param n_trials: number of trials
    :return: significance borderline
    """
    return binom.ppf(1.0 - alpha, n_trials, 0.5) / n_trials
