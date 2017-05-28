import os

import IPython.display as ipd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as skl
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm
import librosa
import librosa.display
import python_utils as utils

plt.rcParams['figure.figsize'] = (17, 5)

# Directory where mp3 are stored.
AUDIO_DIR = os.environ.get('/Volumes/SAMSUNG/James/Physics and Machine Learning/Data/Music')

# Load metadata and features.
meta_data_path = '/Volumes/SAMSUNG/James/Physics and Machine Learning/Data/Music/fma_metadata'
tracks = utils.load(meta_data_path + '/tracks.csv')
genres = utils.load(meta_data_path + '/genres.csv')
features = utils.load(meta_data_path + '/features.csv')
echonest = utils.load(meta_data_path + '/echonest.csv')

np.testing.assert_array_equal(features.index, tracks.index)
assert echonest.index.isin(tracks.index).all()

tracks.shape, genres.shape, features.shape, echonest.shape