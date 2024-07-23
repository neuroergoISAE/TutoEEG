import mne
import numpy as np
import pandas as pd

from pyxdf import load_xdf

import matplotlib.pyplot as plt
import seaborn as sns

from time import time
from mne.preprocessing import ICA
from mne_icalabel import label_components
from meegkit.asr import ASR
from meegkit.utils.matrix import sliding_window


## Utils
def freq_to_band(freq, bands):
    for band_name, (fmin, fmax) in bands.items():
        if fmin <= freq <= fmax:
            return band_name
    return np.nan

def run_ica(data, method, output='iclabel', confidence=.80, fit_params=None, fig_dir=None, verbose=False):
    ica = ICA(
        # n_components=len(data.ch_names),
        n_components=20,
        method=method,
        fit_params=fit_params,
        max_iter="auto",
        random_state=0,
    )
    t0 = time()
    ica.fit(data)
    fit_time = time() - t0
    title = f"ICA decomposition using {method} (took {fit_time:.1f}s)"

    if output == 'source':
        ica.plot_sources(data, title=title)
    elif output == 'component':
        ica.plot_components(title=title)
    elif output == 'iclabel':
        ic_labels = label_components(data, ica, method="iclabel")

        confident_labels = [idx for idx, proba in enumerate(ic_labels['y_pred_proba']) if proba>confidence]
        exclude_idx = [idx for idx, label in enumerate(ic_labels['labels']) if idx in confident_labels and label not in ["brain", "other"]]

        if verbose and len(exclude_idx)>0:
            fig = ica.plot_components(picks=exclude_idx, title=title, show=False)
            fig.savefig(fig_dir)

        reconst_data = data.copy()
        ica.apply(reconst_data, exclude=exclude_idx)
        return reconst_data

def run_asr(data, tmin=0, tmax=30, method='euclid', fit_params=None):
    sfreq = data.info['sfreq']
    n_chans = len(data.ch_names)
    values = data.get_data()

    # Train on a clean portion of data
    asr = ASR(method=method)
    train_idx = np.arange(tmin * sfreq, tmax * sfreq, dtype=int)
    _, sample_mask = asr.fit(values[:, train_idx])

    # Apply filter using sliding (non-overlapping) windows
    X = sliding_window(values, window=int(sfreq), step=int(sfreq))
    Y = np.zeros_like(X)
    for i in range(X.shape[1]):
        Y[:, i, :] = asr.transform(X[:, i, :])

    clean_data = Y.reshape(n_chans, -1)    # reshape to (n_chans, n_times)

    out = mne.io.RawArray(clean_data, data.info)

    return out


def process_speed_caping(fname, data_directory, file_format, biosemi_to_1020_names, bands, artifact_rejection):

    micro_to_volt_ratio = 10**-6 #mne default unit is volt

    ## Loading
    mne.viz.set_browser_backend(backend, verbose=None)

    # If xdf
    if file_format == 'xdf':
        xdf, info = load_xdf(f'{data_directory}\\{fname}.xdf')
        
        # Extract eeg stream and markers for later
        all_markers = []
        for stream in xdf:
            if stream['info']['type'][0] == 'EEG':
                eeg_xdf = stream
            elif stream['info']['type'][0] == 'Markers':
                marker_df = pd.DataFrame(data={'time' : stream['time_stamps'], 'name' : np.array(stream['time_series']).T[0]})
                all_markers.append(marker_df)
                
        all_marker_df = pd.concat(all_markers).sort_values(by='time')
        
        # Extract eeg infos
        ## Timestamps and values
        eeg_values = np.array(eeg_xdf['time_series'])
        eeg_times = eeg_xdf['time_stamps']

        ## Metadata
        biosemi_to_mne_types = {
            'EEG':'eeg',
            'AUX':'misc',
            'EXG':'misc',
            'Trigger':'misc',
        }

        ch_names = [ch['label'][0] for ch in eeg_xdf['info']['desc'][0]['channels'][0]['channel']]
        ch_types = [biosemi_to_mne_types[ch['type'][0]] for ch in eeg_xdf['info']['desc'][0]['channels'][0]['channel']]
        units = set([ch['unit'][0] for ch in eeg_xdf['info']['desc'][0]['channels'][0]['channel']])
        print('Data unit(s) :', units)

        ## Focus on sampling frequency because of my PTSDs
        frequencies = pd.Series(1/np.diff(eeg_times))
        
        # fig, ax = plt.subplots()
        # frequencies.plot(kind="box", vert=False, ax=ax)
        # ax.set_xlim(int(frequencies.min()), int(frequencies.max()))
        # ax.set_xlabel('Sampling Frequency (Hz)')
        # plt.show()

        sfreq = frequencies.mean()

        # Create MNE Raw object
        raw_info = mne.create_info(ch_names = ch_names, sfreq = sfreq, ch_types = ch_types)
        raw_data = eeg_values.T #xdf is (n_samples, n_channels) while mne is (n_channels, n_samples)
        raw_data*= micro_to_volt_ratio
        raw = mne.io.RawArray(raw_data, raw_info)
        
        # Add markers as annotations
        annotations = mne.Annotations(
            onset = all_marker_df['time']-eeg_xdf['time_stamps'][0],  # in seconds, aligned with eeg time serie
            duration=[0]*len(all_marker_df),  # often 0 for us
            description=all_marker_df['name'],
            )
        raw.set_annotations(annotations)

    # If set
    if file_format == 'set':
        raw = mne.io.read_raw_eeglab(f'{data_directory}\\sub-{subject}\\{subject}.set', preload=True)

        unknown_types = {
            'aux': ['AUX1', 'AUX10', 'AUX11', 'AUX12', 'AUX13', 'AUX14', 'AUX15', 'AUX16', 'AUX2', 'AUX3', 'AUX4', 'AUX5', 'AUX6', 'AUX7', 'AUX8', 'AUX9'],
            'exg': ['EX1', 'EX2', 'EX3', 'EX4', 'EX5', 'EX6', 'EX7', 'EX8'],
            'trigger': ['Trig1'],
        }

        unknown_ch = [ch for l in list(unknown_types.values()) for ch in l]
        new_types = ['misc' if ch in unknown_ch else 'eeg' for ch in raw.ch_names]

        raw.set_channel_types(dict(zip(raw.ch_names, new_types)))

    # Montage
    mne.channels.get_builtin_montages()
    standard_1020 = mne.channels.make_standard_montage('standard_1020')

    raw.pick('eeg') # Keep only channels of the montage
    mne.rename_channels(raw.info, biosemi_to_1020_names)

    raw.set_montage(standard_1020)

    ## Markers
    events, original_event_dict = mne.events_from_annotations(raw)
    r_original_event_dict = {value:key for key, value in original_event_dict.items()}

    event_df = pd.DataFrame(data=events, columns=['index', 'duration', 'id'])
    event_df['name'] = event_df['id'].apply(lambda x : r_original_event_dict[x])
    event_df['time'] = event_df['index']/int(raw.info['sfreq'])

    ## Cropping
    tmin = event_df[event_df['name']=='Start']['time'].sort_values().tolist()[0] # At Start trigger
    tmax = event_df[event_df['name']=='D pressed']['time'].sort_values().tolist()[2] + 20 # 20s after 3rd D pressed trigger
    raw.crop(tmin, tmax)

    ## Preprocessing
    clean = raw.copy()
    clean.set_eeg_reference('average')
    clean.filter(1,40)

    ## Advanced Preprocessing
    available_ica = [
        # ('fastica', None), # fast to compute
        ('infomax', dict(extended=True)),   # needed for iclabel
    ]
    fullclean_dict = {'none':clean}
    fullclean_dict['asr'] = run_asr(clean, method='euclid', fit_params=None)
    for method,fit_params in available_ica :
        fullclean_dict[method] = run_ica(clean, method, fit_params=fit_params, fig_dir=f'{result_directory}\\{fname}_{method}.png', verbose=True)

    ## Resting State

    ### Epoching
    triggers = ['EC', 'EO']
    current_event_dict = {key:value for key, value in original_event_dict.items() if key in triggers}

    tmin = -0.5  # start of each epoch 
    tmax = 25  # end of each epoch
    baseline = None  # means from t before to stim onset (t = 0)
    picks = mne.pick_types(clean.info, eeg=True)
    reject = None # this can be highly data dependent

    rs_allepochs = {}
    for method, fullclean in fullclean_dict.items() :
        rs_allepochs[method] = mne.Epochs(fullclean, events, current_event_dict, tmin, tmax, proj=True,
                            picks=picks, baseline=baseline,
                            reject=reject, preload=True)



    ## Auditory Oddball

    ### Epoching
    triggers = ['normal', 'odd']
    current_event_dict = {key:value for key, value in original_event_dict.items() if key in triggers}

    tmin = -0.2  # start of each epoch 
    tmax = 1  # end of each epoch
    baseline = (-0.2, 0)  # means from t before to stim onset (t = 0)
    picks = mne.pick_types(clean.info, eeg=True)
    reject = None # this can be highly data dependent

    oddball_allepochs = {}
    for method, fullclean in fullclean_dict.items() :
        oddball_allepochs[method] = mne.Epochs(fullclean, events, current_event_dict, tmin, tmax, proj=True,
                            picks=picks, baseline=baseline,
                            reject=reject, preload=True)

        if method == artifact_rejection:
            mean_evoked_dict = {condition: oddball_allepochs[method][condition].average() for condition in triggers}
            fig = mean_evoked_dict['odd'].plot_joint(show=False)
            fig.savefig(f'{result_directory}\\{fname}_oddball.png')

    ## Calculation Task

    ### Epoching
    triggers = ['F pressed', 'D pressed']
    rename_triggers = {'F pressed':'easy', 'D pressed':'hard'}
    current_event_dict = {rename_triggers[key]:value for key, value in original_event_dict.items() if key in triggers}

    tmin = -0.5  # start of each epoch 
    tmax = 5  # end of each epoch
    baseline = None  # means from t before to stim onset (t = 0)
    picks = mne.pick_types(clean.info, eeg=True)
    reject = None # this can be highly data dependent

    calculation_allepochs = {}
    for method, fullclean in fullclean_dict.items() :
        calculation_allepochs[method] = mne.Epochs(fullclean, events, current_event_dict, tmin, tmax, proj=True,
                            picks=picks, baseline=baseline,
                            reject=reject, preload=True)



    ### Frequency Bands
    triggers = ['easy', 'hard']
    compute_method = 'welch'
    calculation_allpsd = {method:calculation_epochs.compute_psd(method=compute_method, fmin=2.0, fmax=40.0) for method, calculation_epochs in calculation_allepochs.items()}


    calculation_allpsd_allbanddf = []
    for method, calculation_psd in calculation_allpsd.items():
        calculation_psd_df = calculation_psd.to_data_frame(long_format=True).drop(columns=['ch_type'])
        calculation_psd_df['band'] = calculation_psd_df['freq'].apply(freq_to_band, bands=bands)
        calculation_band_df = calculation_psd_df.copy().dropna().drop(columns=['freq']).groupby(by=['condition', 'epoch', 'channel', 'band'], observed=True).mean().reset_index()
        calculation_band_df['method'] = method
        calculation_allpsd_allbanddf.append(calculation_band_df)

    calculation_allpsd_banddf = pd.concat(calculation_allpsd_allbanddf)


    ## Speed Caping

    oddball_mean_df = oddball_allepochs[artifact_rejection]
    oddball_mean_df = oddball_mean_df.to_data_frame().groupby(by=['condition', 'time']).mean().drop(columns=['epoch']).reset_index().melt(id_vars=['condition', 'time'], var_name='channel')
    oddball_mean_df = oddball_mean_df.pivot(columns='condition', index=['time', 'channel'], values='value').reset_index()
    oddball_mean_df.columns = oddball_mean_df.columns.to_list()

    oddball_mean_df['contrast'] = oddball_mean_df['odd'] - oddball_mean_df['normal']

    calculation_band_df = calculation_allpsd_banddf[calculation_allpsd_banddf['method']==artifact_rejection]

    return oddball_mean_df, calculation_band_df



## Params
data_directory = '..\\Data\\XDF'
result_directory = '..\\Results'

teams = {
    1:'SCASE',
    2:'Michel',
    3:'Py',
    4:'JahanP',
}


ch_name_df = pd.read_csv('..\\Data\\biosemi_channels.csv')
biosemi_to_1020_names = dict(zip(ch_name_df['Alphabetical'], ch_name_df['1020']))

bands = {
    'theta': (4, 8), 
    'alpha': (8, 12),
    'lowbeta ': (12, 20), 
    'highbeta': (20, 30), 
    'gamma': (30, 40)
    }

backend = 'matplotlib'

file_format = 'xdf'
artifact_rejection = 'infomax'

for team_number, team_name in teams.items():
    print(f'Processing team {team_name}')
    fname = f'sub-P00{team_number}_ses-S001_task-Default_run-001_eeg_{team_name}'

    oddball_mean_df, calculation_band_df = process_speed_caping(fname=fname, data_directory=data_directory, file_format=file_format, biosemi_to_1020_names=biosemi_to_1020_names, bands=bands, artifact_rejection=artifact_rejection)
    
    oddball_mean_df['team_number'] = team_number
    oddball_mean_df['team_name'] = team_name
    oddball_mean_df.to_csv(f"{result_directory}\\oddball_team{team_number}.csv", index=False)

    calculation_band_df['team_number'] = team_number
    calculation_band_df['team_name'] = team_name
    calculation_band_df.to_csv(f"{result_directory}\\calculation_team{team_number}.csv", index=False)