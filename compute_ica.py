import os

import autoreject
import matplotlib
import mne
from matplotlib import pyplot as plt

from common import make_dirs
from config import RESULTS_PATH, DATASETS
from datasets import Dataset

matplotlib.use('Agg')


def compute_ica(dataset: Dataset, exp_n: int, output_path: str):
    make_dirs(output_path)

    #%% read referenced and high-pass filtered (1Hz) data
    raw = dataset.read_data(exp_n=exp_n)

    #%% extract concept trials
    use_events = ['animal_image', 'tool_image']
    events, event_dict = mne.events_from_annotations(raw)
    epochs = mne.Epochs(
        raw, events, event_id={e: event_dict[e] for e in use_events},
        tmin=-1, tmax=dataset.concept_trial_period_tmax,
        baseline=None, decim=4, detrend=None, preload=True, verbose=True
    )

    #%% exclude bad concept trials (with artifacts)
    epochs.drop(dataset.exp_info[exp_n]['bad_concept_trials'])

    #%% additionally automatically exclude epochs to improve ICA fit (this is not really necessary but just for a sanity check)
    ar = autoreject.AutoReject(n_interpolate=[1, 2, 3, 4], random_state=42, n_jobs=8, verbose=True)
    epochs_ar, reject_log = ar.fit_transform(epochs, return_log=True)

    fig = reject_log.plot('horizontal')
    fig.savefig(os.path.join(output_path, f'{exp_n}_epochs_reject_log.png'))
    plt.close(fig)

    good_epochs = epochs[~reject_log.bad_epochs]

    #%% compute ICA on concept trials
    ica = mne.preprocessing.ica.ICA(method='fastica', max_iter=10000).fit(good_epochs)

    #%% save IC components
    ica.save(os.path.join(output_path, f'{exp_n}_ica.fif'))

    #%% plot the first 15 IC components
    fig = ica.plot_components(range(15))
    fig.savefig(os.path.join(output_path, f'{exp_n}_ica_components.png'))
    plt.close(fig)


# compute ICs for all datasets
for dataset_name, dataset_info in DATASETS.items():
    for exp_n in dataset_info['cls'].exp_info.keys():
        compute_ica(
            dataset=dataset_info['cls'](dataset_bids_path=dataset_info['path']),
            exp_n=exp_n,
            output_path=os.path.join(RESULTS_PATH, f'ICA_{dataset_name}')
        )
