import matplotlib
matplotlib.use('pgf')

from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('seaborn-paper')

params = {
    'axes.labelsize': 10,
    'font.size': 10,
    'legend.fontsize': 10,
    'legend.title_fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': 'Times New Roman',
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
dataset_names = ['Data1', 'Data2']
datasets = []
dfs = []

for dataset_name in dataset_names:
    dataset = DATASETS[dataset_name]['cls'](dataset_bids_path=DATASETS[dataset_name]['path'])
    datasets.append(dataset)

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
        'RandomForest': 'RF (CV)',
    }

    table = []
    for res in results:
        for rr in res['results']:
            row = res['info'].copy()
            row.update(rr)

            # significant accuracy?
            row['significant'] = row['score'] > borderline_significance_level(alpha=0.05, n_trials=row['samples'])

            row['score'] *= 100
            row['event_title'] = titles[row['event_name']]
            row['classifier_title'] = classifier_names[row['classifier_name']]
            table.append(row)
    df = pd.DataFrame(table)
    dfs.append(df)


#%%
event_order = [
    'image',
    'silent_naming_task', 'visual_imagery_task', 'auditory_imagery_task', 'tactile_imagery_task',
]

nrows = 1
ncols = 2
figsize = (6.3 * 1, 2.5 * nrows)
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=True,
                         gridspec_kw={'width_ratios': [5, 2]},
                         sharey='row'
                         )

for dataset_name, dataset, df, ax in zip(dataset_names, datasets, dfs, axes):

    df_selection = df[
        (df['event_name'].isin(event_order)) &
        (df['channel'] != 'all')
    ]

    df_selection_group = df_selection.groupby(['event_title', 'classifier_title', 'exp_n'])

    df_selection_sum = df_selection_group['significant'].sum()
    df_selection_group_n_channels = df_selection_group['significant'].count()

    df_selection = (df_selection_sum / df_selection_group_n_channels * 100.0).reset_index()

    # print
    print('----')
    pd.set_option("display.max_rows", None, "display.max_columns", None, 'display.precision', 1)
    print(df_selection.groupby(['event_title', 'classifier_title'])['significant'].describe())
    print('---')

    sns_event_order = [event_name for event_name in titles.values() if
                       event_name in df_selection['event_title'].unique()]
    sns_kwargs = dict(data=df_selection, x='event_title', y='significant', hue='classifier_title',
                      dodge=True, palette='colorblind',
                      order=sns_event_order,
                      hue_order=list(classifier_names.values()),
                      ax=ax)

    sns.boxplot(**sns_kwargs,
                notch=False,
                linewidth=0.5,
                showfliers=False,
                fliersize=1,
                # saturation=1,
                width=0.7,
                medianprops={
                    'linewidth': 0.5,
                    'solid_capstyle': 'butt'
                },
                # showmeans=True
                )

    # sns.swarmplot(**sns_kwargs, size=3, edgecolor='black', linewidth=0.2)

    h, l = ax.get_legend_handles_labels()
    # remove swarmplot from the legend
    ax.legend(#title='Classifier',
              ncol=2, loc='best', handles=h, labels=l,
              # bbox_to_anchor=(1, 1.2)
              # columnspacing=1,
              # handletextpad=0.5,
              handlelength=1.5,
              handleheight=0.5,
              # borderaxespad=0.0
              )
    if dataset_name == 'Data2':
        ax.get_legend().remove()

    ax.set_xlabel(None)
    if dataset_name == 'Data1':
        ax.set_ylabel('Percentage of significant channels')
    else:
        ax.set_ylabel(None)
    ax.set_title(f'Dataset {dataset_name[-1]}')

    ax.tick_params(axis='x', direction='out', length=0)
    ax.tick_params(axis='y', direction='out', length=0)

    ax.set_axisbelow(True)
    ax.grid(axis='y', color='0.9', linestyle='--', linewidth=0.7)

sns.despine(top=True, right=True, left=True, bottom=True)
fig.tight_layout()

output_path = results_path + '_plots'
make_dirs(output_path)
fig.savefig(os.path.join(output_path, 'P2_AGG_boxplot_singleChannel_percentage.pdf'), bbox_inches='tight')
plt.close(fig)
