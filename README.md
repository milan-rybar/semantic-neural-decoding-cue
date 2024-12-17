# Using data from cue presentations results in grossly overestimating semantic BCI performance

Source code for the paper "Using data from cue presentations results in grossly overestimating semantic BCI performance", [https://doi.org/10.1038/s41598-024-79309-y](https://doi.org/10.1038/s41598-024-79309-y)

### Data and Results paths

Change the following variables in `config.py`:
- `RESULTS_PATH`: A path where to store the results.
- `DATASETS`: Paths for the two datasets in BIDS format. We will make the two datasets publicly available soon and add links to them here then.

###  Analyses

- `full_period_analysis.py`: An analysis using all time points of corresponding regions of interest. 
- `sliding_window.py`: An analysis using a sliding window approach.
- `replicate_Simanova_analysis.py`: An analysis based on a study by Simanova _et al_ (2010).
- `replicate_Murphy_analysis.py`: An analysis based on a study by Murphy _et al_ (2011).

Scripts with suffix names `*_plot_*` plot the corresponding analysis results.

Analyses are implemented for parallel execution over the set of parameters.
This is achieved by the [jug](https://jug.readthedocs.io/en/latest/) package.
Run it by 
```console
jug execute [analysis script name].py
```
and see the progress by
```console
jug status [analysis script name].py
```

Alternatively, this parallel execution can be removed by commenting out a decorator `@TaskGenerator` before `run_analysis` method in each analysis script. For example:
```python
#@TaskGenerator
def run_analysis(...):
```


### ICAs

Precomputed ICAs are stored in `ICA_Data1` and `ICA_Data2` directories for Datasets 1 and 2, respectively. The directories also contain visualization of the first 15 IC components for each participant. The ICAs and these plots were created by `compute_ica.py`.


### Bootstrapping simulation

The statistical thresholds for mean classification accuracies for each dataset by a bootstrapping simulation were computed by `compute_mean_acc.py`. 


### Conda environment

Create the Conda environment from the `environment.yml` file:
```console
conda env create -f environment.yml
```
Activate the new environment:
```console
conda activate semantic-neural-decoding-cue
```
