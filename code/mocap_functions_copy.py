from scipy.signal import argrelextrema
import sys
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import signal
import neo.core as neo
import matplotlib.pyplot as plt
import elephant 
import quantities as pq
import spike_train_functions
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit
import torch
from torch import nn
import torch.nn.functional as F
import multiprocessing
from joblib import Parallel, delayed
import pickle
import torchvision
import itertools
scaler = StandardScaler()
num_cores = multiprocessing.cpu_count()
from tqdm.auto import tqdm

#Simple feedforward ANN for decoding kinematics
class model_ann(nn.Module):
    def __init__(self, input_size, output_size, layer_size, batch_size=100):
        super(model_ann, self).__init__()
        self.input_size,  self.layer_size, self.output_size, self.batch_size = input_size, layer_size, output_size, batch_size
        
        #List layer sizes
        self.layer_hidden = np.concatenate([[input_size], layer_size, [output_size]])
        
        #Compile layers into lists
        self.layer_list = nn.ModuleList(
            [nn.Linear(in_features=self.layer_hidden[idx], out_features=self.layer_hidden[idx+1]) for idx in range(len(self.layer_hidden)-1)] )        
 
    def forward(self, x):
        #Encoding step
        for idx in range(len(self.layer_list)):
            x = F.tanh(self.layer_list[idx](x))

        return x

#Simple feedforward ANN for decoding kinematics
class model_cnn(nn.Module):
    def __init__(self, input_size, output_size, layer_size, batch_size=100):
        super(model_ann, self).__init__()
        self.input_size,  self.layer_size, self.output_size, self.batch_size = input_size, layer_size, output_size, batch_size
        
        #List layer sizes
        self.conv = torch.nn.Conv2d(input_size,layer_size)
        self.layer_hidden = np.concatenate([[layer_size], layer_size, [output_size]])
        
        #Compile layers into lists
        self.layer_list = nn.ModuleList(input_size, layer_size
            [self.conv]+[nn.Linear(in_features=self.layer_hidden[idx], out_features=self.layer_hidden[idx+1]) for idx in range(len(self.layer_hidden)-1)] )   
        def forward(self, x):
            #Encoding step
            for idx in range(len(self.layer_list)):
                x = F.tanh(self.layer_list[idx](x))

            return x

#Helper function to pytorch train networks for decoding
def train_model(model, optimizer, criterion, max_epochs, training_dataset, device, print_freq=10):
    print("Begun training")
    train_loss_array = []
    model.train()
    # Loop over epochs
    for epoch in range(max_epochs):
        train_batch_loss = []
        print("Epoch: " + str(epoch))
        progress_bar = tqdm(range(0, len(training_dataset), model.batch_size))
        for b in range(0, len(training_dataset), model.batch_size):
            batch_x, batch_y = training_dataset[b:b+model.batch_size]

            optimizer.zero_grad() # Clears existing gradients from previous epoch
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            output = model(batch_x)
            train_loss = criterion(output[:,-1,:], batch_y)
            train_loss.backward() # Does backpropagation and calculates gradients
            optimizer.step() # Updates the weights accordingly

            train_batch_loss.append(train_loss.item())
            progress_bar.update(1)
        print('*',end='')
        train_loss_array.append(train_batch_loss)
        #Print Loss
        if (epoch+1)%print_freq == 0:
            print('')
            print('Epoch: {}/{} ...'.format(epoch+1, max_epochs), end=' ')
            print('Train Loss: ' + str(np.mean(train_batch_loss)))
    return train_loss_array

#Helper function to pytorch train networks for decoding
def train_validate_model(model, optimizer, criterion, max_epochs, training_dataset, validation_dataset, device, print_freq=10, early_stop=20):
    print("Begun training, validating")
    
    train_loss_array = []
    validation_loss_array = []
    # Loop over epochs
    min_validation_loss, min_validation_std, min_validation_counter, min_validation_epoch = np.inf, np.inf, 0, 0
    for epoch in range(max_epochs):
        #___Train model___
        print("Epoch: " + str(epoch))
        progress_bar = tqdm(range(0, len(training_dataset), model.batch_size))
        model.train()
        train_batch_loss = []
        validation_batch_loss = []
        for b in range(0, len(training_dataset), model.batch_size):
            batch_x, batch_y = training_dataset[b:b+model.batch_size]

            optimizer.zero_grad() # Clears existing gradients from previous epoch
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            output = model(batch_x)
            train_loss = criterion(output[:,-1,:], batch_y)
            train_loss.backward() # Does backpropagation and calculates gradients
            optimizer.step() # Updates the weights accordingly

            train_batch_loss.append(train_loss.item())
            progress_bar.update(1)
        
        train_loss_array.append(train_batch_loss)

        #___Evaluate Model___
        with torch.no_grad():
            model.eval()
            print("Evaluating")
            #Generate train set predictions
            val_bar = tqdm(range(0, len(validation_dataset), model.batch_size))
            for b in range(0, len(validation_dataset), model.batch_size):
                batch_x, batch_y = validation_dataset[b:b+model.batch_size]

                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                output = model(batch_x)
                validation_loss = criterion(output[:,-1,:], batch_y)

                validation_batch_loss.append(validation_loss.item())
                val_bar.update(1)

        validation_loss_array.append(validation_batch_loss)

        #Compute average loss on batch
        train_epoch_loss = np.mean(train_batch_loss)
        train_epoch_std = np.std(train_batch_loss)
        validation_epoch_loss = np.mean(validation_batch_loss)
        validation_epoch_std = np.std(validation_batch_loss)

       #Check if validation loss reaches minimum 
        if validation_epoch_loss < min_validation_loss:
            print('*',end='')
            min_validation_loss = np.copy(validation_epoch_loss)
            min_validation_std = np.copy(validation_epoch_std)
            min_validation_counter = 0
            min_validation_epoch = np.copy(epoch+1)

            min_train_loss = np.copy(train_epoch_loss)
            min_train_std = np.copy(train_epoch_std)
            
        else:
            print('.',end='')
            min_validation_counter += 1

        #Print Loss Scores
        if (epoch+1)%print_freq == 0:
            print('')
            print('Epoch: {}/{} ...'.format(epoch+1, max_epochs), end=' ')
            print('Train Loss: {:.4f}  ... Validation Loss: {:.4f}'.format(train_epoch_loss,validation_epoch_loss))
        
        #Early stop if no validation improvement over set number of epochs
        if min_validation_counter > early_stop:
            print(' Early Stop; Min Epoch: {}'.format(min_validation_epoch))
            break

    loss_dict = {'min_validation_loss':min_validation_loss, 'min_validation_std':min_validation_std,'min_validation_epoch':min_validation_epoch, 
    'min_train_loss':min_train_loss, 'min_train_std':min_train_std,
    'train_loss_array':train_loss_array, 'validation_loss_array':validation_loss_array, 'max_epochs':max_epochs}
    return loss_dict


#Helper function to pytorch train networks for decoding
def train_validate_test_model(model, optimizer, criterion, max_epochs, training_dataset, validation_dataset, testing_dataset,device, print_freq=10, early_stop=20):
    print("Begun training, validating, testing")
    train_loss_array = []
    validation_loss_array = []
    test_loss_array = []
    # Loop over epochs
    min_validation_loss, min_validation_std, min_validation_counter, min_validation_epoch = np.inf, np.inf, 0, 0
    for epoch in range(max_epochs):
        #___Train model___
        model.train()
        train_batch_loss = []
        validation_batch_loss = []
        test_batch_loss = []
        for b in range(0, len(training_dataset), model.batch_size):
            batch_x, batch_y = training_dataset[b:b+model.batch_size]
            optimizer.zero_grad() # Clears existing gradients from previous epoch
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            output = model(batch_x)
            train_loss = criterion(output[:,-1,:], batch_y)
            train_loss.backward() # Does backpropagation and calculates gradients
            optimizer.step() # Updates the weights accordingly

            train_batch_loss.append(train_loss.item())
        
        train_loss_array.append(train_batch_loss)

        #___Evaluate Model___
        with torch.no_grad():
            model.eval()
            #Generate validation set predictions
            for b in range(0, len(validation_dataset), model.batch_size):
                batch_x, batch_y = validation_dataset[b:b+model.batch_size]
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                output = model(batch_x)
                validation_loss = criterion(output[:,-1,:], batch_y)

                validation_batch_loss.append(validation_loss.item())

            validation_loss_array.append(validation_batch_loss)

            #Generate test set predictions
            for b in range(0, len(testing_dataset), model.batch_size):
                batch_x, batch_y = testing_dataset[b:b+model.batch_size]
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                output = model(batch_x)
                test_loss = criterion(output[:,-1,:], batch_y)

                test_batch_loss.append(test_loss.item())

            test_loss_array.append(test_batch_loss)

        #Compute average loss on batch
        train_epoch_loss = np.mean(train_batch_loss)
        train_epoch_std = np.std(train_batch_loss)
        validation_epoch_loss = np.mean(validation_batch_loss)
        validation_epoch_std = np.std(validation_batch_loss)
        test_epoch_loss = np.mean(test_batch_loss)
        test_epoch_std = np.std(test_batch_loss)

       #Check if validation loss reaches minimum 
        if validation_epoch_loss < min_validation_loss:
            print('*',end='')
            min_validation_loss = np.copy(validation_epoch_loss)
            min_validation_std = np.copy(validation_epoch_std)
            min_validation_counter = 0
            min_validation_epoch = np.copy(epoch+1)

            min_train_loss = np.copy(train_epoch_loss)
            min_train_std = np.copy(train_epoch_std)
            min_test_loss = np.copy(test_epoch_loss)
            min_test_std = np.copy(test_epoch_std)


        else:
            print('.',end='')
            min_validation_counter += 1

        #Print Loss Scores
        if (epoch+1)%print_freq == 0:
            print('')
            print('Epoch: {}/{} ...'.format(epoch+1, max_epochs), end=' ')
            print('Train Loss: {:.2f}  ... Validation Loss: {:.2f} ... Test Loss: {:.2f}'.format(train_epoch_loss, validation_epoch_loss, test_epoch_loss))
        
        #Early stop if no validation improvement over set number of epochs
        if min_validation_counter > early_stop:
            print(' Early Stop; Min Epoch: {}'.format(min_validation_epoch))
            break

    loss_dict = {'min_validation_loss':min_validation_loss, 'min_validation_std':min_validation_std,'min_validation_epoch':min_validation_epoch, 
    'min_train_loss':min_train_loss, 'min_train_std':min_train_std,
    'min_test_loss':min_test_loss, 'min_test_std':min_test_std,
    'train_loss_array':train_loss_array, 'validation_loss_array':validation_loss_array, 'test_loss_array':test_loss_array, 'max_epochs':max_epochs}
    return loss_dict


#Dataset class to handle mocap dataframes from SEE project
class SEE_Dataset(torch.utils.data.Dataset):
    #'Characterizes a dataset for PyTorch'
    def __init__(self, cv_dict, fold, partition, kinematic_df, neural_df, offset, window_size, data_step_size, device, kinematic_type='posData', scale_data=True, flip_outputs=False):
        #'Initialization'
        self.cv_dict = cv_dict
        self.fold = fold
        self.flip_outputs = flip_outputs
        self.partition = partition
        self.trial_idx = cv_dict[fold][partition]
        self.num_trials = len(self.trial_idx) 
        self.offset = offset
        self.window_size = window_size
        self.data_step_size = data_step_size
        self.device = device
        self.posData_list, self.neuralData_list = self.process_dfs(kinematic_df, neural_df)
        if scale_data:
            self.posData_list = self.transform_data(self.posData_list)
            self.neuralData_list = self.transform_data(self.neuralData_list)

        self.kinematic_type = kinematic_type
        self.split_offset = np.round(self.offset/self.data_step_size).astype(int)

        self.X_tensor, self.y_tensor = self.load_splits()
        self.num_samples = np.sum(self.X_tensor.size(0))

    def __len__(self):
        #'Denotes the total number of samples'
        return self.num_samples

    def __getitem__(self, slice_index):
        if self.flip_outputs:
            return self.y_tensor[slice_index,:,:], self.X_tensor[slice_index,:,:]
        else:
            return self.X_tensor[slice_index,:,:], self.y_tensor[slice_index,:,:]

    #**add functionality to separate eye, object, and body markers
    def process_dfs(self, kinematic_df, neural_df):
        posData_list, neuralData_list = [], []
        for trial in self.trial_idx:
            posData_array = np.stack(kinematic_df['posData'][kinematic_df['trial'] == trial].values).transpose() 
            neuralData_array = np.stack(neural_df['rates'][neural_df['trial'] == trial].values).squeeze().transpose() 

            posData_list.append(posData_array)
            neuralData_list.append(neuralData_array)

        return posData_list, neuralData_list

    def format_splits(self, data_list):
        data_tensor = torch.from_numpy(
            np.concatenate(
                [np.pad(data_list[trial], ((self.window_size,self.window_size),(0,0)), mode='constant') for trial in range(self.num_trials)]
                )  
            ).unfold(0, self.window_size, self.data_step_size).transpose(1,2)

        return data_tensor
    
    def load_splits(self):
        y_tensor = self.format_splits(self.neuralData_list)

        if self.kinematic_type == 'posData':
            X_tensor = self.format_splits(self.posData_list)
        # elif self.kinematic_type == 'both':
        #     y1 = self.format_splits(self.rotData_list)
        #     y2 = self.format_splits(self.posData_list)
        #     y_tensor = torch.stack([y1, y2], dim=2)

        X_tensor, y_tensor = X_tensor[:-self.split_offset,::self.data_step_size,:], y_tensor[self.split_offset:,::self.data_step_size,:]
        assert X_tensor.shape[0] == y_tensor.shape[0]
        return X_tensor, y_tensor

    #Zero mean and unit std
    def transform_data(self, data_list):
        #Iterate over trials and apply normalization
        # np.mean(np.concatenate(data_list),0)
        # np.std(np.concatenate(data_list),0)
        scaled_data_list = []
        for data_trial in data_list:
            scaled_data_trial = scaler.fit_transform(data_trial)
            scaled_data_list.append(scaled_data_trial)

        return scaled_data_list

class Video_Dataset(torch.utils.data.Dataset):
    def __init__(self, cv_dict,  fold, partition, video_path, video_df, neural_df, subsample_scalar=1):
        self.video_path = video_path
        self.reader = torchvision.io.VideoReader(video_path, "video")
        self.md = self.reader.get_metadata()
        self.subsample_scalar = subsample_scalar
        self.video_df = video_df
        self.neural_df = neural_df
        self.cv_dict = cv_dict
        self.fold = fold
        self.partition = partition
        self.trial_idx = cv_dict[fold][partition]
        self.video_datalist, self.neural_datalist = self.process_dfs(video_df, neural_df)
        # Make this more robust
        self.aspect_ratio = (-(1056 // -self.subsample_scalar), -(1440 // -self.subsample_scalar))
        self.y_tensor = self.scale_y_data()
        self.x_tensor = torch.tensor(np.concatenate(self.video_datalist, axis=0))

        self.x_size = self[0][0].shape[2]
        self.y_size = self.y_tensor.shape[1]
        # super().__init__()
    
    '''
    Generates scaled y data
    '''
    def scale_y_data(self):
        scaler = StandardScaler()
        y_tensor = torch.tensor(scaler.fit_transform(np.concatenate(self.neural_datalist, axis=0)))
        return y_tensor

    def process_dfs(self, video_df, neural_df):
        videoData_list, neuralData_list = [], []
        for trial in self.trial_idx:
            # TODO: I'm throwing away the last trial because it spills over the end of the video by several seconds. Look for a more elegant fix!
            if trial != 217:
                videoData_array = np.stack(video_df['timeStamps'][video_df['trial'] == trial].values).transpose() 
                neuralData_array = np.stack(neural_df['rates'][neural_df['trial'] == trial].values).squeeze().transpose() 

                videoData_list.append(videoData_array)
                neuralData_list.append(neuralData_array)

        return videoData_list, neuralData_list
    '''
    Allows time indexing (s) into videos, returns relevant frames. Note that 
    indexes must be floats.

    Returns list(np.array)
    '''
    def __getitem__(self, key):
        frames = []
        
        for time_stamp in self.x_tensor[key]:
            try:
                time_stamp = float(time_stamp.numpy())
                self.reader.seek(time_stamp)
                frame = np.moveaxis(np.array(next(self.reader)['data']), 0, 2)[::self.subsample_scalar,::self.subsample_scalar,0] / 255.0
                frame = np.reshape(frame, (1,-1))
                frames.append(frame)
            except:
                # TODO: FIX UNDERLYING ISSUE!
                print("Warning: an exception occured finding time_stamp: " + str(float(time_stamp.numpy())))
                frame = np.random.rand(1, self.x_size)
                frames.append(frame)

        
        return torch.tensor(frames), self.y_tensor[key,:]


    def __len__(self):
        return self.y_tensor.shape[0]

    """
    Given frames, print them
    """
    def print_frames(self, frames, step=1):
        num_frames = len(frames)
        if num_frames > 1:
            fig, axes = plt.subplots(nrows=1, ncols= -(num_frames // -step), figsize=(24,24))
            i=0
            for n in range(0, num_frames, step):
                #curr_ax = axes[i//8, i%8] 
                curr_ax = axes[i] 
                curr_ax.imshow(np.reshape(frames[n], self.aspect_ratio))
                i+=1
            return fig
        else:
            plt.imshow(np.reshape(frames[0], self.aspect_ratio))
    
    """
    Given time stamp(s), prints the frames at those timestamps
    """
    def print_frame_at_time(self, time_stamp, step=1):
        time_stamp = float(time_stamp)
        self.reader.seek(time_stamp)
        frames = [np.moveaxis(np.array(next(self.reader)['data']), 0, 2)[::self.subsample_scalar,::self.subsample_scalar,0] / 255.0]
        fig = self.print_frames(frames, step)
        return fig
    
    
# Utility function to load dataframes of preprocessed kinematic/neural data
def load_mocap_df(data_path):
    kinematic_df = pd.read_pickle(data_path + 'kinematic_df.pkl')
    neural_df = pd.read_pickle(data_path + 'neural_df.pkl')
    video_df = pd.read_pickle(data_path + 'video_df.pkl')

    # read python dict back from the file
    metadata_file = open(data_path + 'metadata.pkl', 'rb')
    metadata = pickle.load(metadata_file)
    metadata_file.close()

    return kinematic_df, neural_df, video_df, metadata


#Vectorized correlation coefficient of two matrices on specified dimension
def matrix_corr(x, y, axis=0):
    num_tpts, _ = np.shape(x)
    mean_x, mean_y = np.tile(np.mean(x, axis=axis), [num_tpts,1]), np.tile(np.mean(y, axis=axis), [num_tpts,1])
    corr = np.sum(np.multiply((x-mean_x), (y-mean_y)), axis=axis) / np.sqrt(np.multiply( np.sum((x-mean_x)**2, axis=axis), np.sum((y-mean_y)**2, axis=axis) ))
    return corr

#Helper function to evaluate decoding performance on a trained model
def evaluate_model(model, dataset, device):
    #Run model through test set
    with torch.no_grad():
        model.eval()
        #Generate train set predictions
        y_pred_tensor = torch.zeros(len(dataset), dataset.y_tensor.shape[1])
        batch_idx = 0
        for b in range(0, len(dataset), model.batch_size):
            batch_x, batch_y = dataset[b:b+model.batch_size]
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            output = model(batch_x)
            y_pred_tensor[batch_idx:(batch_idx+output.size(0)),:] = output[:,-1,:]
            batch_idx += output.size(0)

    y_pred = y_pred_tensor.detach().cpu().numpy()
    return y_pred