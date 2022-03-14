from scipy.io import wavfile
import numpy as np
import glob
import os


def filter(data):
    r = 4
    a = np.ones(r)  # parameters of AR model 
    for n, sample in enumerate(data[r:], r):
        # prediction for sample n, as linear combination of r previous samples
        y_pred_n = np.sum(a*data[n-r:n][::-1])  # a[0]*y[n-1] + a[1]*y[n-2] + ...

    return 2*data


for filepath in glob.iglob('wav/*.wav'):
    # Read audio data
    samplerate, data = wavfile.read(filepath)

    new_data = filter(data)

    # Write data without impulse noise to file
    basename = os.path.basename(filepath)
    new_filepath = os.path.join("smoothed", os.path.splitext(basename)[0] + "_smoothed.wav")
    print(new_filepath)
    wavfile.write(new_filepath, samplerate, new_data)



