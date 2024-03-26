import os

import mne
import numpy as np
from jug import TaskGenerator
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score, ParameterGrid
from sklearn.pipeline import make_pipeline

from common import unique_dict_id, make_dirs, pickle_dict
from config import RESULTS_PATH, DATASETS, CLASSIFIERS_OPTIONS


@TaskGenerator
def run_analysis(dataset_name: str, exp_n: int, output_dirname: str, classifier_name: str, event_names: list,
                 component_order: str, csp_n_components: list):
    _results = locals().copy()  # for storing all arguments
    results_hash = unique_dict_id(_results)

    dataset = DATASETS[dataset_name]['cls'](dataset_bids_path=DATASETS[dataset_name]['path'])

    #%% read referenced and high-pass filtered (1Hz) data
    raw = dataset.read_data(exp_n=exp_n)

    #%% load pre-trained ICA (on subset of concept trials)
    ica = mne.preprocessing.ica.read_ica(os.path.join(RESULTS_PATH, f'ICA_{dataset_name}', f'{exp_n}_ica.fif'))
    assert ica.n_components_ == raw.info['nchan']

    #%% suppress eye blinks by ICA
    ica.apply(raw, exclude=[dataset.exp_info[exp_n]['ica_eye_blink']])

    #%% filter
    raw.filter(4.1, None)
    raw.filter(None, 18.3)

    #%%
    results = []
    for event_name in event_names:
        #%% extract epochs
        decimate = 16  # 2048 to 128 Hz
        use_events = [f'animal_{event_name}', f'tool_{event_name}']
        events, event_dict = mne.events_from_annotations(raw)
        epochs = mne.Epochs(
            raw, events, event_id={e: event_dict[e] for e in use_events},
            tmin=0.095, tmax=0.36,
            baseline=None, decim=decimate, detrend=None, preload=True, verbose=True
        )

        #%% exclude epochs from bad concept trials
        epochs.drop(dataset.exp_info[exp_n]['bad_concept_trials'])
        assert len(epochs) == dataset.n_trials - len(dataset.exp_info[exp_n]['bad_concept_trials'])

        #%% prepare data
        x = np.vstack([
            epochs[use_events[0]].get_data(),
            epochs[use_events[1]].get_data()
        ])
        y = np.hstack([
            [0] * epochs[use_events[0]].get_data().shape[0],
            [1] * epochs[use_events[1]].get_data().shape[0]
        ])
        assert x.shape[0] == y.shape[0]

        _results['samples'] = len(y)  # store for later significance computations
        n_channels = len(epochs.ch_names)

        #%%
        cv = StratifiedKFold(n_splits=15, shuffle=False)

        for n_components in csp_n_components:
            clf = make_pipeline(
                # reduce dimensions to have matrices with full ranks
                mne.decoding.UnsupervisedSpatialFilter(PCA(n_components=n_channels - 1), average=False),
                mne.decoding.CSP(n_components=n_components, component_order=component_order),
                CLASSIFIERS_OPTIONS[classifier_name]
            )

            scores = cross_val_score(clf, x, y, cv=cv, error_score='raise')
            results.append(dict(score=np.mean(scores), event_name=event_name,
                                n_components=n_components,
                                classifier_name=classifier_name,
                                all_scores=scores))

    #%%save
    output_path = os.path.join(RESULTS_PATH, output_dirname)
    make_dirs(output_path)
    pickle_dict(os.path.join(output_path, '{}.pickle'.format(results_hash)),
                dict(info=_results, results=results))


params_grid = dict(
    classifier_name=list(CLASSIFIERS_OPTIONS.keys()),
    component_order=['mutual_info', 'alternate']
)

for params in ParameterGrid(params_grid):
    for dataset_name, dataset_info in DATASETS.items():
        for exp_n in dataset_info['cls'].exp_info.keys():
            run_analysis(
                dataset_name=dataset_name,
                exp_n=exp_n,
                output_dirname=f'replicate_Murphy_{dataset_name}',
                csp_n_components=[2, 4, 6, 8, 10],
                event_names=['image'] + dataset_info['cls'].tasks,
                **params
            )
