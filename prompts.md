What is the best approach to accomplish the following goal?

---

I have more than 100 EEG datasets in the BIDS format.

I want to load them from disk because the data is too large and does not fit in RAM.

I want to load 10 sec of segment from randomly chosen subject at a time.

So, if batch size is 64, 64 segments should be load from various subjects and timepoints.

Could you write python code to make batches (in PyTorch) from BIDS dataset in a fast and shuffled way?

1 segment from 1 subject at a time, please.

Handle multiple datasets, please.

Please downsample to 128 Hz.

Please adjust differences in channels and montages.

I have 32 CPU threads. So, please parallelize the loading step.

Preparing all the BIDS layouts will be useful for training?

I suppose channels and sampling rate is consistent in a dataset.

I think edf format is too slow to load. 
So, save array as numpy or something and channels and sampling rate as a meta data.
