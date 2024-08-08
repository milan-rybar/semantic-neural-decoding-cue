import matplotlib
matplotlib.use('Agg')

import mne
import numpy as np

from config import DATASETS


def compute_stat_acc(dataset_name, participant, alpha):
    dataset = DATASETS[dataset_name]['cls'](dataset_bids_path=DATASETS[dataset_name]['path'])

    #%% prepare ground truth
    ground_truth = {}

    for exp_n in [participant]:
        raw = dataset.read_bids_data(exp_n=exp_n)

        # extract semantic categories
        use_events = ['animal_image', 'tool_image']
        events, event_dict = mne.events_from_annotations(raw)

        categories = events[(events[:, 2] == event_dict[use_events[0]]) | (events[:, 2] == event_dict[use_events[1]])]
        assert categories.shape == (dataset.n_trials, 3)

        # exclude bad concept trials
        categories = np.delete(categories[:, 2], dataset.exp_info[exp_n]['bad_concept_trials'])
        assert len(categories) == dataset.n_trials - len(dataset.exp_info[exp_n]['bad_concept_trials'])

        trials_y = np.empty((len(categories, )))
        trials_y[categories == event_dict[use_events[0]]] = 0  # 0 for animals
        trials_y[categories == event_dict[use_events[1]]] = 1  # 1 for tools

        ground_truth[exp_n] = trials_y


    #%% bootstrap accuracies
    mean_accuracies = []
    for _ in range(10**6):
        # mean acc of each participant
        accuracies = []
        for exp_n, truth in ground_truth.items():
            predictions = np.random.randint(low=0, high=2, size=len(truth))
            exp_acc = np.mean(predictions == truth)

            accuracies.append(exp_acc)

        mean_accuracies.extend(accuracies)

    #%%
    # for alpha in [5, 1, 0.1, 0.01, 0.001, 0.0001]:
    threshold = np.percentile(mean_accuracies, 100 - alpha) * 100
    return threshold


results = {}
dataset_name = 'Data1'
for exp_n in DATASETS[dataset_name]['cls'].exp_info.keys():
    threshold = compute_stat_acc(dataset_name=dataset_name, participant=exp_n, alpha=5)
    results[exp_n] = threshold

print(results)

# Dataset 1
# {1: 56.44171779141104, 2: 56.70731707317073, 3: 56.17283950617284, 4: 56.64335664335665, 5: 56.60377358490566, 6: 56.666666666666664, 7: 56.547619047619044, 9: 56.95364238410596, 10: 56.41025641025641, 11: 56.36363636363636, 12: 56.95364238410596}

# Dataset 2
# {1: 55.158730158730165, 2: 55.158730158730165, 3: 55.158730158730165, 4: 55.158730158730165, 5: 55.158730158730165, 6: 55.158730158730165, 7: 55.158730158730165}
