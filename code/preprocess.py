import sys
sys.path.append('/home/moasi/SEE/SEE_decoding/code') 
import mocap_functions
import numpy as np
import pandas as pd
import neo
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import signal
from scipy.interpolate import interp1d
import spike_train_functions
import elephant
import quantities as pq
# import h5py
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch as torch
from torch import nn
import torch.nn.functional as F
from joblib import Parallel, delayed
import multiprocessing
import Neural_Decoding
import pickle
import seaborn as sns
sns.set()
sns.set_style("white")


def load_raw_data(pos_fnames, eyes_path, video_path):
    cam_df = pd.read_csv(pos_fnames)

    # Subtract wrist marker from fingers
    forelimb_markers = ['carpal', 'radiusProx', 'radiusDistal', 'ulnarProx', 'ulnarDistal']
    wrist_marker = 'carpal'
    forelimb_mask = np.logical_or.reduce([cam_df.columns.str.contains(pat=pat) for pat in forelimb_markers])
    finger_mask = np.invert(forelimb_mask)

    for marker_name in cam_df.columns[finger_mask]:
        if '_x' in marker_name:
            cam_df[marker_name] = cam_df[marker_name] - cam_df[f'{wrist_marker}_x']
        elif '_y' in marker_name:
            cam_df[marker_name] = cam_df[marker_name] - cam_df[f'{wrist_marker}_y']
            
    # Calculate grip aperture
    cam_df['grip_aperture'] = np.sqrt(
        np.square(cam_df['thumbDistal_x'] - cam_df['indexProx_x']) + 
        np.square(cam_df['thumbDistal_y'] - cam_df['indexProx_y']))

    # Load eyes and concatenate marker information into column names from first 3 rows
    eyes_df = pd.read_csv(eyes_path, header=[1,2,3]).iloc[:,1:]
    eyes_df.columns = ['_'.join(eyes_df.columns[idx]) for idx in range(len(eyes_df.columns))]

    # Add columns from eyes_df
    cam_df = pd.concat([cam_df, eyes_df], axis=1)

    pos_mask = np.logical_or.reduce((cam_df.columns.str.contains(pat='_x'), cam_df.columns.str.contains(pat='_y'), 
                                    cam_df.columns.str.contains(pat='_z')))
    marker_names = cam_df.columns[pos_mask].values.tolist()

    #===========================================================================
    # Inspect proportion of NaNs where tracking lost
    null_percent = cam_df.isnull().astype(int).sum().values / len(cam_df)
    sort_indices = np.argsort(null_percent)[::-1]

    for name, val in zip(cam_df.columns[sort_indices], null_percent[sort_indices]):
        print(f'| {name} {np.round(val, 2)}', end=' ')
    #===========================================================================
    # Set threshold for likelihood
    score_threshold = 0.5

    # Pull out marker names stripped of suffix (only markers have scores and likelihood DLC variables)
    score_mask = cam_df.columns.str.contains(pat='_x')
    score_names = cam_df.columns[score_mask].values
    marker_names_stripped = cam_df.columns[score_mask].str.split('_').str[:-1].str.join('_').values

    # marker_pos_names indicates which vars are stored in kinematic_df. Only append _x, _y, _z markers
    marker_pos_names = list()
    for mrk_name in marker_names_stripped:
        # Separate likelihood and position data for 
        mrk_score_mask = np.logical_and(cam_df.columns.str.contains(pat=mrk_name), 
                                        np.logical_or(cam_df.columns.str.contains(pat='error'), 
                                                    cam_df.columns.str.contains(pat='likelihood')))
        mrk_pos_mask = np.logical_and(cam_df.columns.str.contains(pat=mrk_name), 
                                        np.logical_or.reduce((cam_df.columns.str.contains(pat='_x'), 
                                                            cam_df.columns.str.contains(pat='_y'),
                                                            cam_df.columns.str.contains(pat='_z'))))   
        # There should only be 1 likelihood variable, and 2 or 3 position variables
        assert np.sum(mrk_score_mask) == 1
        assert np.sum(mrk_pos_mask) == 2 or np.sum(mrk_pos_mask) == 3
        mrk_score_name = cam_df.columns[mrk_score_mask].values[0]
        pos_name_list = cam_df.columns[mrk_pos_mask].values.tolist()
        marker_pos_names.extend(pos_name_list)

        score_data = cam_df[mrk_score_name].values
        threshold_mask = score_data < score_threshold

        # Set position data values to zero if likelihood below threshold
        for pos_name in pos_name_list:
            pos_data = cam_df[pos_name]
            pos_data[threshold_mask] = 0.0

            # Remove nans (need to test with interpolation)
            pos_data = np.nan_to_num(pos_data, copy=True, nan=0.0)
            
            # Update dataframe
            cam_df[pos_name] = pos_data
    #===========================================================================
    cb_dict = dict()
    unit_idx = 0
    for cb_idx in range(1,3):
        #Use neo module to load blackrock files
        fpath = '../data/SPK20220308/task_data/'
        # TODO: Fix this
        if cb_idx == 1:
            experiment_dict = sio.loadmat(f'{fpath}eventsCB{cb_idx}_corrected.mat')
        else:
            experiment_dict = sio.loadmat(f'{fpath}SPKRH20220308_CB{cb_idx}_ev_explicit.mat')
        nev = neo.io.BlackrockIO(f'{fpath}SPKRH20220308_CB{cb_idx}_quiver4toyPKPK4Rotation_delay_001_RETVR_DSXII_corrected.nev')
        ns2 = neo.io.BlackrockIO(f'{fpath}SPKRH20220308_CB{cb_idx}_quiver4toyPKPK4Rotation_delay_001.ns2')

        sampling_rate_list = ns2.header['signal_channels'][['name','sampling_rate']]
        sampling_rate = 30000
        analog_sampling_rate = 1000
        eye_sampling_rate = 500
        camera_sampling_rate = 40

        #nev seg holds spike train information to extract
        nev_seg = nev.read_segment()
        tstart = nev_seg.t_start.item()
        tstop = nev_seg.t_stop.item()

        #Group spiketrain timestamps by unit id
        unit_timestamps = dict()
        for st in nev_seg.spiketrains:
            if st.annotations['unit_id'] == 1:
                unit_timestamps[unit_idx] = st.times
                unit_idx += 1

        #Grab indeces for camera frames
        cam_trigger = ns2.get_analogsignal_chunk(channel_names=['FlirCam']).transpose()[0]
        num_analog_samples = len(cam_trigger)
        trigger_val = 18000 #threshold where rising edge aligns frame, may need to tweak
        cam_frames = np.flatnonzero((cam_trigger[:-1] < trigger_val) & (cam_trigger[1:] > trigger_val))+1

        cb_dict[f'cb{cb_idx}'] = {'tstart': tstart, 'tstop': tstop, 'unit_timestamps': unit_timestamps,
                                'cam_frames': cam_frames, 'experiment_dict': experiment_dict}
    #===========================================================================
    experiment_dict = cb_dict['cb1']['experiment_dict']
    cam_frames =  cb_dict['cb1']['cam_frames']

    #Load variables from struct (struct indexing is unfortunately hideous)
    #ev_ex = experiment_dict['df']
    ev_ex = experiment_dict['eventsCB1']
    gocON = ev_ex['gocON'][0][0][0]
    #gocOFF = ev_ex['gocOFF'][0][0][0]
    #stmv = ev_ex['stmv'][0][0][0]
    contact = ev_ex['contact_C'][0][0][0]
    end_hold = ev_ex['endhold_C'][0][0][0]
    layout = ev_ex['LAYOUT_C'][0][0][0] # Inserts layout into dataframe
    position = ev_ex['POSITION_C'][0][0][0]
    #===========================================================================
    #Define game event for alignment, and window around marker
    event_ts = end_hold
    #event_ts = event_ts[event_ts < 1000] # events less than 1000 seconds for short DLC tracking
    e_start, e_stop = [0,2]
    num_events = len(event_ts)
    #===========================================================================
    # Find scale/timeshift between CB1 and CB2
    cb2_align_ts = cb_dict['cb2']['experiment_dict']['df']['gocON'][0][0][0]
    assert len(cb2_align_ts) == len(gocON) 
    cb2_start, cb2_end = cb2_align_ts[0], cb2_align_ts[-1]

    ts_shift = gocON[0] - cb2_start 
    ts_scale = (cb2_end - cb2_start) / (gocON[-1] -  gocON[0])
    #===========================================================================
    unit_timestamps = cb_dict['cb1']['unit_timestamps'].copy()

    # Shift and scale time stamps between the two machines
    unit_timestamps_cb2 = cb_dict['cb2']['unit_timestamps'].copy()
    unit_timestamps_cb2_corrected = dict()
    for unit_idx, unit_ts in unit_timestamps_cb2.items():
        ts_corrected = (unit_ts + ts_shift * (pq.s)) / (ts_scale * pq.s)
        unit_timestamps_cb2_corrected[unit_idx] = ts_corrected


    unit_timestamps.update(unit_timestamps_cb2_corrected)

    #===========================================================================
    #Append convolved firing rates to dataframe
    kernel_halfwidth = 0.250 #in seconds
    kernel = elephant.kernels.RectangularKernel(sigma=kernel_halfwidth/np.sqrt(3)*pq.s)  #***** Change if you want Carlos
    sampling_period = 0.01*pq.s
    window_length = int((e_stop-e_start)/(sampling_period).item())
        
    #List to store neural data
    rate_col = list()
    unit_col = list()
    trial_col_neural = list()
    layout_col_neural = list()
    position_col_neural = list()

    #List to store kinematic data
    posData_col = list()
    name_col = list()
    trial_col_kinematic = list()
    layout_col_kinematic = list()
    position_col_kinematic = list()

    #List to store video data
    timeStamps_col = list()
    trial_col_video = list()
    layout_col_video = list()
    position_col_video = list()

    kinematic_metadata = dict()
    neural_metadata = dict()
    for e_idx, e_ts in enumerate(event_ts):
        print(e_idx, end=' ')
        
        #Load kinematic data
        #Identify which frames fall in the time window
        frame_mask = np.logical_and(cam_frames > ((e_ts+e_start) * analog_sampling_rate), cam_frames < ((e_ts+e_stop) * analog_sampling_rate))
        frame_idx = np.flatnonzero(frame_mask) #Pull out indeces of valid frames
        kinematic_metadata[e_idx] = {'time_data':np.linspace(e_ts+e_start, e_ts+e_stop, window_length)}
        
        for mkr in marker_pos_names:
            marker_pos = cam_df[mkr].values[frame_idx]
            f = interp1d(np.linspace(0,1,marker_pos.size), marker_pos)
            marker_interp = f(np.linspace(0,1,window_length))
            
            posData_col.append(marker_interp)
            name_col.append(mkr)
            trial_col_kinematic.append(e_idx)
            layout_col_kinematic.append(layout[e_idx])
            
            position_col_kinematic.append(position[e_idx])
        
        #Load neural data
        # use as its own 2d matrix
        unit_rate = list()
        for unit_idx, unit_ts in unit_timestamps.items():
            rate = spike_train_functions.spike_train_rates(unit_ts, e_ts+e_start, e_ts+e_stop, sampling_rate, kernel, sampling_period).transpose()
            
            rate_col.append(rate)
            trial_col_neural.append(e_idx)
            layout_col_neural.append(layout[e_idx])
            unit_col.append(unit_idx)
            neural_metadata[e_idx] = {'time_data':frame_idx}
            position_col_neural.append(position[e_idx])
        # TODO: Ask about this
        # all_rates.append(unit_rate)

        timeStamps_col.append(np.linspace(e_ts+e_start, e_ts+e_stop, window_length))
        trial_col_video.append(e_idx)
        layout_col_video.append(layout[e_idx])
        position_col_video.append(position[e_idx])
            
    #Pickle convolved rates
    neural_dict = {'rates':rate_col, 'unit':unit_col, 'trial':trial_col_neural, 'layout': layout_col_neural, 'position':position_col_neural}
    neural_df = pd.DataFrame(neural_dict)
    neural_df['count'] = neural_df['rates'].apply(np.sum)


    kinematic_dict = {'name':name_col, 'posData':posData_col, 'trial':trial_col_kinematic, 'layout': layout_col_kinematic, 'position':position_col_kinematic}
    kinematic_df = pd.DataFrame(kinematic_dict)

    video_dict = {'timeStamps':timeStamps_col, 'trial': trial_col_video, 'layout': layout_col_video, 'position': position_col_video}
    video_df = pd.DataFrame(video_dict)

    metadata={'kinematic_metadata':kinematic_metadata, 'neural_metadata':neural_metadata, 'num_trials':num_events, 'kernel_halfwidth':kernel_halfwidth}
    #===========================================================================
    #Save metadata
    output = open(fpath + 'metadata.pkl', 'wb')
    pickle.dump(metadata, output)
    output.close()
    #===========================================================================
    kinematic_df, neural_df, metadata = mocap_functions.load_mocap_df('../data/SPK20220308/task_data/')

    sampling_rate = 100
    kernel_offset = int(metadata['kernel_halfwidth'] * sampling_rate)  #Convolution kernel centered at zero, add to neural offset
    neural_offset = 20
    offset = neural_offset + kernel_offset
    window_size = 20
    data_step_size = 1 

    num_trials = metadata['num_trials']

    #===========================================================================

    #Generate cv_dict for regular train/test/validate split (no rolling window)
    cv_split = ShuffleSplit(n_splits=5, test_size=.25, random_state=0)
    val_split = ShuffleSplit(n_splits=1, test_size=.25, random_state=0)
    cv_dict = {}
    for fold, (train_val_idx, test_idx) in enumerate(cv_split.split(np.arange(num_trials))):
        for t_idx, v_idx in val_split.split(train_val_idx): #No looping, just used to split train/validation sets
            cv_dict[fold] = {'train_idx':train_val_idx[t_idx], 
                            'test_idx':test_idx, 
                            'validation_idx':train_val_idx[v_idx]} 
    return cv_dict, fold, video_path, video_df, neural_df
    #===========================================================================