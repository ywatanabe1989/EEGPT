from glob import glob
import os
import mne
from bids import BIDSLayout
import mngs
from tqdm import tqdm
import numpy as np
    
# specify the root directory where the EEG files are located

how can i parallelize it?
def get_edf_files(bids_roots):
    layouts = []
    for bids_root in tqdm(bids_roots):
        try:
            layout = BIDSLayout(bids_root, validate=True)
            layouts.append(layout)
        except Exception as e:
            print(e)
    layouts = np.hstack([list(layout.files) for layout in layouts])
    edf_files = np.hstack(mngs.gen.search(".edf", list(layouts))[1])
    return edf_files

    
    
    # for dirpath, dirnames, filenames in os.walk(bids_root):
        
    #     try:
    #         filename = filenames[0] # assume all files in a dataset have the same channels and sampling rate
    #         # for filename in filenames:
    #         # check if the file is an .edf file
    #         if filename.endswith('.edf'):
    #             # construct the full file path
    #             full_path = os.path.join(dirpath, filename)
    #             print('File Path:', full_path)

    #             # load the data using mne.io.read_raw_edf
    #             raw = mne.io.read_raw_edf(full_path, preload=True, verbose=False)

    #             # print the channel names
    #             print('Channel Names:', raw.ch_names)

    #             # print the sampling rate
    #             print('Sampling Rate:', raw.info['sfreq'])

    #             print('---' * 20)
    #     except Exception as e:
    #         print(e)

    
if __name__ == "__main__":
    bids_roots = glob("./data/EEGPT_datasets/ds*")
    edf_files = get_layouts(bids_roots)
    mngs.io.save(edf_files, "./res/edf_files.npy")
    mngs.io.load("./res/edf_files.npy")
