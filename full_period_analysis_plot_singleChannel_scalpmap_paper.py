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
    'pdf.fonttype': 42,
    # 'savefig.pad_inches': 0,
}
plt.rcParams.update(params)

import os
from glob import glob
import pandas as pd

from config import RESULTS_PATH, DATASETS
from common import make_dirs, unpickle_dict, borderline_significance_level


#%%
dataset_name = 'Data2'
dataset = DATASETS[dataset_name]['cls'](dataset_bids_path=DATASETS[dataset_name]['path'])
results_path = os.path.join(RESULTS_PATH, f'full_period_{dataset_name}')
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
        # TODO: temporally to fix the incorrect directory name for the sliding window approach
        if 'time_end' in rr:
            continue
        # TODO: ignore image on Dataset 2 with only 600ms
        if rr['event_name'] == 'image' and res['info']['epoch_length'] != dataset.image_length:
            continue

        row = res['info'].copy()
        row.update(rr)

        # significant accuracy?
        row['significant'] = row['score'] > borderline_significance_level(alpha=0.05, n_trials=row['samples'])

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
event_order = [
    'image',
    # 'silent_naming_task', 'visual_imagery_task', 'auditory_imagery_task', 'tactile_imagery_task',
]

nrows = 1
ncols = len(df['classifier_name'].unique())

figsize = (1.8 * ncols, 1.8 * nrows)
fig = plt.figure(  # nrows=nrows, ncols=ncols,
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

# col_idx = 0
row_idx = 0
for col_idx, classifier_name in enumerate(sorted(df['classifier_name'].unique())):
    ax = axes[col_idx]

    df_selection = df[
        (df['event_name'] == 'image') &
        (df['classifier_name'] == classifier_name) &
        (df['channel'] != 'all')
    ]
    # vmax = 62.251655629139066  # the same vmax for both datasets
    vmax = 66

    df_mean = df_selection.groupby(['channel'])['score'].mean().reset_index()

    ch_acc = []
    for ch_name in exp_info.ch_names:
        accuracy = df_mean[(df_mean['channel'] == ch_name)]['score'].iloc[0]
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
        res=512
    )
    ax.set_title(f'{classifier_names[classifier_name]}')

    # plt.colorbar(im, ax=ax, label='Accuracy')

cbar = grid.cbar_axes[0].colorbar(im)
cbar.ax.set_ylabel('Accuracy')

# fig.tight_layout()

output_path = results_path + '_plots'
make_dirs(output_path)
fig.savefig(os.path.join(output_path, 'P_boxplot_singleChannel_scalpmap.pdf'), bbox_inches='tight')
plt.close(fig)
