import mne
from mne_bids import BIDSPath, read_raw_bids

from common import high_pass


class Dataset(object):

    name = None
    concept_trial_period_tmax = None
    tasks = None
    task_length = None
    image_length = None
    n_trials = None
    exp_info = None
    mean_acc_threshold = None

    def __init__(self, dataset_bids_path: str):
        self.dataset_bids_path = dataset_bids_path

    def read_bids_data(self, exp_n: int):
        #%% read dataset from BIDS format
        datatype = 'eeg'
        subject = '{:0>2d}'.format(exp_n)
        task = 'eeg'
        suffix = 'eeg'

        bids_path = BIDSPath(subject=subject, task=task, suffix=suffix,
                             datatype=datatype, root=self.dataset_bids_path)
        return read_raw_bids(bids_path=bids_path, verbose=True)

    def read_data(self, exp_n: int):
        #%% read dataset from BIDS format
        raw = self.read_bids_data(exp_n=exp_n)
        raw.load_data()

        #%% set montage
        self.set_montage(raw)

        #%% reference data
        raw.set_eeg_reference(ref_channels=['EXG1', 'EXG2'])  # mean of ear's references

        #%% use only EEG channels excluding bad channels
        raw = raw.pick_channels([channel for channel in raw.ch_names[:64]
                                 if channel not in self.exp_info[exp_n]['bad_channels']])

        #%% high-pass filter with 1 Hz cutoff
        high_pass(raw, high_pass_cut=1, high_pass_order=4)

        #%%
        return raw

    @classmethod
    def set_montage(cls, raw):
        montage = mne.channels.make_standard_montage('biosemi64')
        raw.set_montage(montage, on_missing='warn')


class Dataset1(Dataset):

    concept_trial_period_tmax = 14.3  # 0.6 + 0.6 + 0.5 + 4 * 3 + 3 * 0.2
    tasks = [
        'silent_naming_task',
        'visual_imagery_task',
        'auditory_imagery_task',
        'tactile_imagery_task'
    ]
    task_length = 3
    image_length = 0.6
    n_trials = 180

    # note: `bad_concept_trials` (numbered from 0)
    exp_info = {
        1: dict(
            # CP6? FT8?
            bad_channels=['Iz', 'TP8', 'P6', 'POz', 'C1'],
            bad_concept_trials=[50, 64, 67, 69, 84, 86, 87, 88, 103, 118, 128, 140, 141, 148, 156, 163, 166],
            ica_eye_blink=0
        ),
        # ? exclude ? different signal after the half of the experiment, hurting electrodes (notes)?
        # all frontal electrodes are affected, the rest could be ok
        2: dict(
            # F6? T7? CP6?
            bad_channels=['TP7', 'CP5', 'P1', 'P7', 'PO3', 'FC6', 'C6', 'O2', 'P9', 'Oz', 'O1', 'C3', 'C5', 'FC5',
                          'PO8', 'P8'],
            bad_concept_trials=[48, 71, 76, 78, 79, 80, 81, 82, 108, 116, 117, 121, 128, 155, 172, 176],
            ica_eye_blink=0
        ),
        # Iz, FT8, TP8? Iz, F2 /// PO8 CP3, TP7
        # quite some artifacts in some channels
        3: dict(
            bad_channels=['TP8', 'CP6', 'Iz', 'Oz'],
            bad_concept_trials=[0, 18, 20, 21, 24, 26, 28, 48, 50, 51, 63, 72, 85, 86, 96, 127, 155, 165],
            ica_eye_blink=0
        ),
        4: dict(
            # a lot of movements and artifacts, deal on trials basis
            bad_channels=['POz', 'P2', 'CP6', 'PO8', 'P4', 'P10', 'C5', 'Iz', 'Pz', 'FC6', 'FT8', 'FC2'],
            bad_concept_trials=[0, 11, 17, 24, 26, 29, 41, 42, 59, 67, 68, 72, 73, 80, 82, 83, 86, 102, 103, 104, 105,
                                106, 107, 115, 116, 117, 119, 125, 131, 140, 148, 149, 156, 161, 167, 174, 175],
            ica_eye_blink=0
        ),
        5: dict(
            # Iz, POz, PO3 - 50 Hz noise, --- P3?
            bad_channels=['Iz', 'CP1', 'P5'],
            bad_concept_trials=[0, 6, 8, 9, 12, 41, 48, 51, 52, 63, 68, 74, 81, 84, 89, 90, 106, 151, 161, 165, 176],
            ica_eye_blink=0
        ),
        6: dict(
            bad_channels=['Iz'],  # there are few completely bad trials (over all channels) that must be excluded
            bad_concept_trials=[13, 24, 36, 45, 47, 48, 59, 60, 61, 78, 87, 91, 96, 97, 98, 107, 108, 122, 127, 131,
                                143, 144, 147, 148, 156, 158, 160, 161, 167, 179],
            ica_eye_blink=0
        ),
        7: dict(
            bad_channels=['FT8', 'FC6', 'C6', 'P10', 'F2'],
            bad_concept_trials=[7, 15, 27, 45, 53, 64, 72, 77, 92, 125, 127, 130],
            ica_eye_blink=0
        ),
        # 8: dict(
        #     # ? exclude ? too many movements and artifacts
        #     # exclude - after eye blinks suppression ERP not clearly visible for P, PO electrodes to represent anything on the screen
        #     # P2, T8 bad after the half of the experiment; P5?
        #     bad_channels=['T8', 'P2'],  # a lot of movements and artifacts, deal on trials basis
        #     bad_concept_trials=[10, 24, 28, 34, 36, 37, 39, 40, 41, 42, 48, 50, 52, 54, 55, 58, 59, 60, 61, 65, 67, 71, 75, 76, 77, 78, 79, 80, 81, 83, 84, 89, 95, 98, 100, 103, 104, 109, 115, 116, 119, 120, 123, 124, 128, 130, 131, 132, 138, 140, 141, 143, 144, 145, 147, 148, 151, 152, 156, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 170, 171, 172, 175, 176, 179]
        # ),
        9: dict(
            bad_channels=['Iz', 'P2'],  # Iz bad at the end; P2 noisy at the beginning
            bad_concept_trials=[0, 22, 23, 24, 34, 39, 47, 54, 60, 64, 72, 92, 93, 99, 100, 101, 105, 108, 114, 120,
                                124, 130, 131, 132, 133, 149, 151, 152, 162],
            ica_eye_blink=0
        ),
        10: dict(
            # PO7 sometimes; do not exclude for ICA, shouldn't make much difference
            bad_channels=['TP7'],
            bad_concept_trials=[0, 5, 10, 18, 19, 28, 33, 48, 58, 60, 63, 71, 73, 79, 82, 87, 96, 101, 119, 131, 146,
                                156, 169, 175],
            ica_eye_blink=0
        ),
        11: dict(
            bad_channels=[],
            bad_concept_trials=[8, 20, 24, 60, 64, 65, 97, 99, 102, 144, 145, 146, 147, 155, 157],
            ica_eye_blink=0
        ),
        12: dict(
            # F7, FT7?
            bad_channels=['Fp1', 'AF7', 'F7', 'FT7'],
            bad_concept_trials=[12, 15, 16, 24, 27, 33, 35, 44, 60, 61, 69, 80, 84, 86, 92, 96, 100, 123, 131, 132, 135,
                                136, 145, 148, 152, 154, 166, 168, 175],
            ica_eye_blink=0
        )
    }

    mean_acc_threshold = {
        0.05: 56.60377358490566,
        0.01: 59.333333333333336,
        0.001: 62.251655629139066
    }

    # for p-value = 0.05
    individual_acc_threshold = {1: 56.44171779141104, 2: 56.70731707317073, 3: 56.17283950617284, 4: 56.64335664335665,
                                5: 56.60377358490566, 6: 56.666666666666664, 7: 56.547619047619044,
                                9: 56.95364238410596, 10: 56.41025641025641, 11: 56.36363636363636,
                                12: 56.95364238410596}


    @classmethod
    def set_montage(cls, raw):
        """
        Update raw dataset according to biosemi64 montage:
        - rename channels
        - set montage
        - set modality type

        :param raw: raw data
        :type raw: `mne.io.base.BaseRaw`
        """
        n_channels = 64  # biosemi 64 channels

        # keep EEG type only for electrodes and 2 references
        raw.set_channel_types({name: 'misc' for name in raw.ch_names[n_channels + 2:-1]})  # except EXG1, EXG2, stimulus
        raw.set_channel_types({'Resp': 'resp'})

        # create montage
        montage = mne.channels.make_standard_montage('biosemi64')
        assert len(montage.ch_names) == n_channels

        # rename channels
        channels_mapping = {o: n for o, n in zip(raw.info['ch_names'][:n_channels], montage.ch_names[:n_channels])}
        raw.rename_channels(channels_mapping)

        # set montage
        raw.set_montage(montage, on_missing='warn')


class Dataset2(Dataset):

    concept_trial_period_tmax = 7.1  # 1 + 0.6 + 0.5 + 5
    tasks = [
        'auditory_imagery_task'
    ]
    task_length = 5
    image_length = 1
    n_trials = 252

    # note: `bad_concept_trials` (numbered from 0)
    exp_info = {
        1: dict(
            bad_channels=['P2'],
            bad_concept_trials=[],
            ica_eye_blink=0
        ),
        2: dict(
            bad_channels=[],
            bad_concept_trials=[],
            ica_eye_blink=0
        ),
        3: dict(
            bad_channels=[],
            bad_concept_trials=[],
            ica_eye_blink=0
        ),
        4: dict(
            bad_channels=[],
            bad_concept_trials=[],
            ica_eye_blink=2
        ),
        5: dict(
            bad_channels=[],
            bad_concept_trials=[],
            ica_eye_blink=0
        ),
        6: dict(
            bad_channels=[],
            bad_concept_trials=[],
            ica_eye_blink=2
        ),
        7: dict(
            bad_channels=[],
            bad_concept_trials=[],
            ica_eye_blink=0
        ),
    }

    mean_acc_threshold = {
        0.05: 55.158730158730165,
        0.01: 57.14285714285714,
        0.001: 59.523809523809526
    }

    # for p-value = 0.05
    individual_acc_threshold = {1: 55.158730158730165, 2: 55.158730158730165, 3: 55.158730158730165,
                                4: 55.158730158730165, 5: 55.158730158730165, 6: 55.158730158730165,
                                7: 55.158730158730165}
