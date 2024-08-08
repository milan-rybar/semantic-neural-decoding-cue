import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import mne
import numpy as np

from config import DATASETS


#%%
dataset_name = 'Data2'
dataset = DATASETS[dataset_name]['cls'](dataset_bids_path=DATASETS[dataset_name]['path'])

#%% prepare ground truth
ground_truth = {}

for exp_n in dataset.exp_info.keys():
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


#%% bootstrap mean accuracies
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
plt.figure()
plt.hist(mean_accuracies)

#%%
for alpha in [5, 1, 0.1, 0.01, 0.001, 0.0001]:
    threshold = np.percentile(mean_accuracies, 100 - alpha) * 100
    print(alpha, alpha / 100.0, '{:.2f}'.format(threshold), threshold)


# Dataset 1
# 5 0.05 56.60 56.60377358490566
# 1 0.01 59.33 59.333333333333336
# 0.1 0.001 62.25 62.251655629139066
# 0.01 0.0001 64.85 64.84848484848484
# 0.001 1e-05 67.26 67.26190476190477
# 0.0001 1e-06 69.14 69.1358025641254

# Dataset 2
# 5 0.05 55.16 55.158730158730165
# 1 0.01 57.14 57.14285714285714
# 0.1 0.001 59.52 59.523809523809526
# 0.01 0.0001 61.51 61.50793650793651
# 0.001 1e-05 63.49 63.49206349206349
# 0.0001 1e-06 64.68 64.68253968253968
