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
    'pgf.texsystem': 'lualatex',  # use lualatex instead of xelatex for memory limitation
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
dataset_name = 'Data2'
dataset = DATASETS[dataset_name]['cls'](dataset_bids_path=DATASETS[dataset_name]['path'])
results_path = os.path.join(RESULTS_PATH, f'replicate_Murphy_{dataset_name}')
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
event_order = [
    'image',
    'silent_naming_task', 'visual_imagery_task', 'auditory_imagery_task', 'tactile_imagery_task',
]

# classifier_name = 'SVM_C1'
# component_order = 'alternate'

nrows = 1  #len(df['classifier_name'].unique())
ncols = 1  #len(df['component_order'].unique())
figsize = (6.3 * ncols, 3.2 * nrows)
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False)

for col_idx, component_order in enumerate(['mutual_info']):

    for row_idx, classifier_title in enumerate([classifier_names['SVM_CV_C']]):
        ax = axes[row_idx, col_idx]

        df_selection = df[
            (df['event_name'].isin(event_order)) &
            (df['classifier_title'] == classifier_title) &
            (df['component_order'] == component_order)
        ]

        # print mean accuracies
        print('----', component_order)
        pd.set_option("display.max_rows", None, "display.max_columns", None, 'display.precision', 1)
        print(df_selection.groupby(['event_name', 'n_components'])['score'].describe())
        print('---')

        sns_event_order = [event_name for event_name in titles.values() if
                       event_name in df_selection['event_title'].unique()]
        sns_kwargs = dict(data=df_selection, x='event_title', y='score', hue='n_components',
                          dodge=True, palette='colorblind',
                          order=sns_event_order,
                          hue_order=list(sorted(df_selection['n_components'].unique())),
                          ax=ax)

        sns.boxplot(**sns_kwargs,
                    notch=False,
                    linewidth=0.5,
                    showfliers=True,
                    fliersize=2,
                    # saturation=1,
                    width=0.7,
                    medianprops={
                        'linewidth': 0.7,
                        'solid_capstyle': 'butt'
                    },
                    showmeans=True
                    )

        sns.swarmplot(**sns_kwargs, size=3, edgecolor='black', linewidth=0.2)

        h, l = ax.get_legend_handles_labels()
        # remove swarmplot from the legend
        ax.legend(title='CSP components', ncol=5, loc='best', handles=h[:len(h) // 2], labels=l[:len(l) // 2],
                  # bbox_to_anchor=(1, 1.2)
                  columnspacing=1,
                  # handletextpad=0.5,
                  handlelength=1.5,
                  handleheight=0.5,
                  # borderaxespad=0.0
                  )

        ax.set_xlabel(None)
        ax.set_ylabel('Classification accuracy')
        ax.set_title(f'{classifier_title}, {component_order.replace("_", "-")}')

        hline_kwargs = dict(color='black', alpha=0.5, linewidth=0.8, zorder=-1000)
        ax.axhline(dataset.mean_acc_threshold[0.05], linestyle='solid', **hline_kwargs)
        ax.axhline(dataset.mean_acc_threshold[0.01], linestyle='dashed', **hline_kwargs)
        ax.axhline(dataset.mean_acc_threshold[0.001], linestyle='dotted', **hline_kwargs)

        ax.tick_params(axis='x', direction='out', length=0)
        ax.tick_params(axis='y', direction='out', length=0)

        ax.set_axisbelow(True)
        ax.grid(axis='y', color='0.9', linestyle='--', linewidth=0.7)

sns.despine(top=True, right=True, left=True, bottom=True)
fig.tight_layout()

output_path = results_path + '_plots'
make_dirs(output_path)
fig.savefig(os.path.join(output_path, f'boxplot_selection.pdf'), bbox_inches='tight')
plt.close(fig)
