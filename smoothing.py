from scipy.io import wavfile
import numpy as np
from numpy import transpose as tr
import glob
import os
from matplotlib import pyplot as plt

def filter(data):
    r = 4 # order of AR model
    theta = np.ones(r)  # vector of parameters of AR model 
    P = np.ones((r, r), dtype=np.float64)
    lambda_f = 0.9  # forgetting factor
    smoothed_data = np.copy(data)
    for n, sample in enumerate(data[r:], r):
        phi = data[n-r:n][::-1]  # r previous samples
        # prediction for sample n, as linear combination of r previous samples
        y_pred_n = np.sum(theta*phi)  # a[0]*y[n-1] + a[1]*y[n-2] + ...

        eps = sample-y_pred_n  # prediction error  

        # New parameters
        k = P @ phi / (lambda_f + tr(phi)@P@phi)
        theta = theta + k*eps
        P = 1/lambda_f * (P - P@phi@tr(phi)*P / (lambda_f + tr(phi)@P@phi))

        # debug
        d1 = np.isnan(theta)
        if True in d1:
            print(f"theta: {theta}")
        d2 = np.isnan(phi)
        if True in d2:
            print(f"phi: {phi}")
        # ~~~~~~

        smoothed_data[n] = y_pred_n # temp


        
        

    return smoothed_data


for filepath in glob.iglob('wav/*.wav'):
    # Read audio data
    samplerate, data = wavfile.read(filepath)

    if filepath not in ['wav\\18.wav']: # debug temp - choose files to process
        print(f"skipp: {filepath}")
        continue
    new_data = filter(data)

    # Write data without impulse noise to file
    basename = os.path.basename(filepath)
    new_filepath = os.path.join("smoothed", os.path.splitext(basename)[0] + "_smoothed.wav")
    wavfile.write(new_filepath, samplerate, new_data)
    print(new_filepath)

    # Plot smoothed
    plt.plot(data)
    plt.plot(new_data)
    plt.legend(["original", "smoothed"])
    plt.show()



