import preprocess
import sys
sys.path.append('/home/moasi/SEE/SEE_decoding/code') 
import mocap_functions_copy
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

def main():
    
    num_cores = multiprocessing.cpu_count()
    scaler = StandardScaler()

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device('cpu')

    torch.backends.cudnn.benchmark = True

    video_path = '../data/SPK20220308/motion_tracking'

    pos_fnames = f'{video_path}/Spike03-08-1557_DLC3D_resnet50_DLCnetwork3D_Spike03-08-1557Sep7shuffle1_500000_AllCam.csv'

    eyes_path = f'{video_path}/SpikeCam5_EYES_03-08-1557DLC_resnet50_DLC-eyesNov4shuffle1_40000_el.csv'
    print("preprocessing begun")
    cv_dict, fold, video_path, video_df, neural_df = preprocess.load_raw_data(pos_fnames, eyes_path, video_path)
    print("preprocessing over")
    batch_size = 1000
    train_params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': num_cores, 'pin_memory':False}
    train_eval_params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': num_cores, 'pin_memory':False}
    validation_params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': num_cores, 'pin_memory':False}
    test_params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': num_cores, 'pin_memory':False}
    
    video_path = "../data/SpikeCam3_03-08-1557.mp4"
    
    training_set = mocap_functions_copy.Video_Dataset(cv_dict,  fold, 'train_idx', video_path, video_df, neural_df, subsample_scalar=10)
    # X_train_data = training_set[:][0][:,-1,:].detach().cpu().numpy()
    # y_train_data = training_set[:][1][:,-1,:].detach().cpu().numpy()

    validation_set = mocap_functions_copy.Video_Dataset(cv_dict,  fold, 'validation_idx', video_path, video_df, neural_df, subsample_scalar=10)

    testing_set = mocap_functions_copy.Video_Dataset(cv_dict,  fold, 'test_idx', video_path, video_df, neural_df, subsample_scalar=10)

    #=============================================================================
    #Define hyperparameters
    lr = 1e-2
    weight_decay = 1e-5
    layer_size=[10, 10]
    max_epochs=3
    input_size = training_set.x_size
    output_size = training_set.y_size

    model_ann = mocap_functions_copy.model_ann(input_size,output_size,layer_size, batch_size=batch_size).to(device)
    # Define Loss, Optimizerints h
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model_ann.parameters(), lr=lr, weight_decay=weight_decay)

    #Train model
    loss_dict = mocap_functions_copy.train_validate_model(model_ann, optimizer, criterion, max_epochs, training_set, validation_set, device, 10, 5)

    #Evaluate trained model
    ann_train_pred = mocap_functions_copy.evaluate_model(model_ann, training_set, device)
    ann_test_pred = mocap_functions_copy.evaluate_model(model_ann, testing_set, device)

    #Compute decoding performance
    y_test_data = testing_set[:][1][:,-1,:].detach().cpu().numpy()
    y_train_data = training_set[:][1][:,-1,:].detach().cpu().numpy()
    ann_train_corr = mocap_functions_copy.matrix_corr(ann_train_pred,y_train_data)
    ann_test_corr = mocap_functions_copy.matrix_corr(ann_test_pred,y_test_data)

    #===========================================================================

        #Testing Data
    start=23
    plt.figure(figsize=(12,8))
    bounds = np.arange(1,2000)
    x_vals = np.arange(len(bounds)) / 100
    for plot_idx, unit_idx in enumerate(range(start, start+4)):
        plt.subplot(2,2,plot_idx+1)
        plt.plot(x_vals,y_test_data[bounds,unit_idx])
        # plt.plot(x_vals,kf_test_pred[bounds,unit_idx])
        plt.plot(x_vals,ann_test_pred[bounds,unit_idx])
        plt.title('Unit ' + str(unit_idx))
        plt.xlabel('Time (s)')
        plt.ylabel('Firing Rate (a.u.)')
        if plot_idx==1:
            plt.legend(['Real', 'KF', 'ANN'], loc=1)
    plt.tight_layout()
    plt.savefig("./results.png")

if __name__ == "__main__":
  main()