import re
import random
from glob import glob
import os

## Functions to sort folder, files in the "natural" way:
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def get_sorted_video_paths(directory):
    video_paths = glob(os.path.join(directory, '*.avi'))
    video_paths.sort(key=natural_keys)
    return video_paths

def prepare_dataset_paths(base_path, train_folder, val_folder):
    train_path = os.path.join(base_path, train_folder)
    val_path = os.path.join(base_path, val_folder)

    train_video_paths = get_sorted_video_paths(train_path)
    val_video_paths = get_sorted_video_paths(val_path)

    return train_video_paths, val_video_paths


def get_indices_by_keyword(idx_list, keyword):
    return [i for i, s in enumerate(idx_list) if keyword in s[0]]

def process_data_indices(idx_list):
    train_indices_n2 = get_indices_by_keyword(idx_list, '/N2_train/')
    val_indices_n2 = get_indices_by_keyword(idx_list, '/N2_val/')

    train_indices_other = get_indices_by_keyword(idx_list, '/unc_train/')
    val_indices_other = get_indices_by_keyword(idx_list, '/unc_val/')

    all_train_idx = train_indices_n2 + train_indices_other
    all_val_idx = val_indices_n2 + val_indices_other

    return all_train_idx, all_val_idx



def get_fps_from_filename(filename):
    start = filename.find('fps_')
    finish = filename.find('nf')
    nfps = filename[start + len('fps_'):finish]
    return int(nfps)

def get_frames_number_from_filename(filename):
    start = filename.find('nf_')
    finish = filename.find('.avi')
    nframes = filename[start + len('nf_'):finish]
    return int(nframes)


def divide_into_time_intervals_idx(nframes, framesxint):
    nintervals = int(nframes / framesxint)
    value = 0
    intervals = []
    for interval in range(0, nintervals):
        ll = value
        value += framesxint
        upl = value - 1
        intervals.append([ll, upl])
    return intervals


def divide_videos_into_subvideos(videos_paths, skipped_frames, seq_length, frames_per_int):
    dataset_list = []  # complete dataset, with subdivision of the original videos
    idx_list = []  # list with the original videos and the index ranges of the corresponding subvideos
    idx = 0  # indexes of the subvideos

    for video in videos_paths:
        video_frames = get_frames_number_from_filename(video)
        video_cap_fps = get_fps_from_filename(video)
        idx_ini = idx
        sample_freq = video_cap_fps / (skipped_frames + 1)  # frequency resulting from sampling

        time_intervals = divide_into_time_intervals_idx(video_frames, frames_per_int)

        for interv in time_intervals:
            start, end = interv
            frame_list = []  # list of frame indexes corresponding to each subvideo
            frames_sampled = 0
            for i in range(start, end + 1, skipped_frames + 1):
                frame_list.append(i)
                frames_sampled += 1
                if frames_sampled == seq_length:  # if video length is reached we save and start sampling another one
                    dataset_list.append([video, frame_list])
                    idx += 1

                    break

        idx_end = idx - 1
        idx_list.append([video, [idx_ini, idx_end], sample_freq])
    return dataset_list, idx_list


def get_max_samp_per_sample(idx_list):
    # calculates how many videos have been generated per video
    videos_per_sample = []
    for video in idx_list:
        nvideos = video[1][1] - video[1][0] + 1
        videos_per_sample.append(nvideos)

    # the minimum is the maximum that can be chosen per video, to have the same number of subvideos for each video.
    max_samp = min(videos_per_sample)
    print(min(videos_per_sample))
    return max_samp


def random_sample_max_subvideos(max_samp, idx_list, all_train_idx, all_val_idx):
    idx_train_list = []
    idx_val_list = []

    # for each video, max_samp subvideos are randomly sampled.
    for idx, video in enumerate(idx_list):
        if idx in all_train_idx:
            try:
                sampled_indexes = random.sample(range(video[1][0], video[1][1] + 1), max_samp)
            except Exception as e:
                sampled_indexes = [*range(video[1][0], video[1][1] + 1, 1)]

            idx_train_list += sampled_indexes

        elif idx in all_val_idx:
            try:
                sampled_indexes = random.sample(range(video[1][0], video[1][1] + 1), max_samp)
            except Exception as e:
                sampled_indexes = [*range(video[1][0], video[1][1] + 1, 1)]

            idx_val_list += sampled_indexes
        else:
            print('error')

    return idx_train_list, idx_val_list




class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True
                
 




