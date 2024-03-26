import matplotlib
# matplotlib.use('pgf')  # for boxplots
matplotlib.use('Agg')  # for scalpmaps

import numpy as np
import mne
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import AxesGrid
from matplotlib import cm
import seaborn as sns
plt.style.use('seaborn-paper')

params = {
    'axes.labelsize': 10,
    'font.size': 10,
    'legend.fontsize': 10,
    'legend.title_fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.titlesize': 10,
    'text.usetex': True,
    'font.family': 'serif',
    # 'font.serif': 'Times New Roman',
    'font.serif': 'Times',
    'pgf.texsystem': 'xelatex',
    'figure.figsize': [6.3, 4],
    'savefig.bbox': 'tight',
    # 'savefig.pad_inches': 0,
}
plt.rcParams.update(params)

import os
from glob import glob
import pandas as pd

from config import RESULTS_PATH, DATASETS
from common import make_dirs, unpickle_dict


#%%
dataset_name = 'Data1'
dataset = DATASETS[dataset_name]['cls'](dataset_bids_path=DATASETS[dataset_name]['path'])
results_path = os.path.join(RESULTS_PATH, f'sliding_window_{dataset_name}')
results = (unpickle_dict(fr) for fr in glob(os.path.join(results_path, '*.pickle')))

#%%
titles = {
    'image': 'Image presentation',
    'silent_naming_task': 'Silent naming',
    'visual_imagery_task': 'Visual imagery',
    'auditory_imagery_task': 'Auditory imagery',
    'tactile_imagery_task': 'Tactile imagery'
}

# split label to 2 multiple lines
titles = {k: '\n'.join(v.split(' ')) for k, v in titles.items()}

classifier_names = {
    'SVM_C1': 'SVM',
    'SVM_CV_C': 'SVM (CV)',
    'LR_L1': 'LR L1',
    'LR_L1_CV': 'LR L1 (CV)',
    'LR_L2': 'LR L2',
    'LDA': 'LDA',
}

table = []
for res in results:
    for rr in res['results']:
        row = res['info'].copy()
        row.update(rr)
        row['score'] *= 100
        row['event_title'] = titles[row['event_name']]
        row['classifier_title'] = classifier_names[row['classifier_name']]
        table.append(row)
df = pd.DataFrame(table)


#%%
params = {
    'axes.titlesize': 10,
    'savefig.pad_inches': 0,  # save as much space as you can!
    'axes.edgecolor': '0.8'  # MNE head outline color (not anymore, hack); colorbar outlines
}
plt.rcParams.update(params)

event_order = [
    'image',
    'silent_naming_task', 'visual_imagery_task', 'auditory_imagery_task', 'tactile_imagery_task',
]

vmin = dataset.mean_acc_threshold[0.05]  # borderline for p-value = 0.05
vmax = dataset.mean_acc_threshold[0.001]  # borderline for p-value = 0.05

n_colors = 256
# original_colors = cm.get_cmap('YlOrRd', n_colors)
original_colors = cm.get_cmap('viridis', n_colors)
# original_colors = cm.get_cmap('Reds', n_colors)
newcolors = original_colors(np.linspace(0, 1, n_colors))
newcolors[0, :] = np.array([1, 1, 1, 1])  # white color
newcmp = ListedColormap(newcolors)

# hacky way to load experiment info :D (a participant with no excluded channels)
_dataset_name = 'Data2'
exp_info = DATASETS[_dataset_name]['cls'](dataset_bids_path=DATASETS[_dataset_name]['path']).read_data(exp_n=7)

#%%
for event_name in ['image']:#df['event_name'].unique():

    _df = df[
        (df['event_name'] == event_name) &
        (df['classifier_name'] == 'SVM_CV_C') &
        (df['time_end'] <= 0.5)
    ]
    vmax = 62.251655629139066  # the same vmax for both datasets

    nrows = len(_df['classifier_name'].unique())
    ncols = len(_df['time_end'].unique())
    figsize = (2 * ncols, 2 * nrows)
    fig = plt.figure(#nrows=nrows, ncols=ncols,
                             figsize=figsize
        # , squeeze=False
    )

    # for placing colorbar directly next to 2 last axes in right column
    grid = AxesGrid(fig, 111,
                    nrows_ncols=(nrows, ncols),
                    axes_pad=(0.02, 0.28),
                    cbar_mode='single',
                    cbar_location='right',
                    cbar_pad=0.1,
                    # share_all=True,
                    # label_mode='L',
                    # cbar_size='3%',
                    )
    axes = grid

    for col_idx, time_end in enumerate(sorted(_df['time_end'].unique())):
        for row_idx, classifier_name in enumerate(sorted(_df['classifier_name'].unique())):
            # ax = axes[row_idx, col_idx]
            ax = axes[col_idx]

            df_selection = _df[
                (_df['event_name'] == event_name) &
                (_df['time_end'] == time_end) &
                (_df['classifier_name'] == classifier_name) &
                (_df['channel'] != 'all')
            ]

            ch_acc = []
            for ch_name in exp_info.ch_names:
                accuracy = df_selection[(df_selection['channel'] == ch_name)]['score'].mean()
                ch_acc.append(accuracy)

            assert len(exp_info.ch_names) == len(ch_acc)

            im, cont = mne.viz.plot_topomap(
                data=ch_acc,
                pos=exp_info.info,
                sensors=True,
                names=None,
                # show_names=False,
                outlines='head',
                border='mean',
                extrapolate='local',
                # vmin=vmin,
                # vmax=vmax,
                vlim=(vmin, vmax),
                cmap=newcmp,
                contours=0,
                mask_params=dict(
                    marker='.', markerfacecolor='k', markeredgecolor='k',
                    linewidth=0, markersize=1),
                axes=ax,
                show=False,
                # outline_line_width=0.5, outline_line_color='0.8', sensor_size=0.05,
                res=256
            )

            if len(df_selection) > 0:
                ax.set_title('{:.1f}-{:.1f} ms'.format(df_selection['time_start'].unique()[0] * 1000,
                                                          df_selection['time_end'].unique()[0] * 1000))
            else:
                ax.set_title(f'{classifier_name}')

            # fig.colorbar(im, ax=ax)


    cbar = grid.cbar_axes[0].colorbar(im)
    cbar.ax.set_ylabel('Accuracy')

    # sns.despine(top=True, right=True, left=True, bottom=True)
    # fig.tight_layout()

    output_path = results_path + '_plots'
    make_dirs(output_path)
    fig.savefig(os.path.join(output_path, f'scalpmap_{event_name}_selection.png'), bbox_inches='tight')
    plt.close(fig)
