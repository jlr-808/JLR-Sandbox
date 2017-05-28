# We'll need numpy for some mathematical operations
import numpy as np
import audioread

# matplotlib for displaying the output
import matplotlib.pyplot as plt
import matplotlib.style as ms
ms.use('seaborn-muted')
%matplotlib inline

from IPython.display import Audio

# and IPython.display for audio output
import IPython.display


# Librosa for audio
import librosa
# And the display module for visualization
import librosa.display

audio_path1 = librosa.util.example_audio_file()
audio_path2 = 'C:/Users/James/Documents/Python Scripts/MIR and DSP/Samples/4649530_Creator__Final_Call__Album_Mix.wav'

y, sr = librosa.load(audio_path1, sr = 44100)
y_me , sr_me = librosa.load(audio_path2,duration = 5.0)

y.shape
y_me.shape
 
# plots 
plt.plot(y_me)
plt.plot(y)

# audio sample
fs = 44100
Audio(y)

# Let's make and display a mel-scaled power (energy-squared) spectrogram
S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

# Convert to log scale (dB). We'll use the peak power as reference.
log_S = librosa.logamplitude(S, ref_power=np.max)

# Make a new figure
plt.figure(figsize=(12,4))

# Display the spectrogram on a mel scale
# sample rate and hop length parameters are used to render the time axis
librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')

# Put a descriptive title on the plot
plt.title('mel power spectrogram')

# draw a color bar
plt.colorbar(format='%+02.0f dB')

# Make the figure layout compact
plt.tight_layout()

plt.plot?