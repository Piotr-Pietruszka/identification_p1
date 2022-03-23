from scipy.io import wavfile
import numpy as np
from numpy import transpose as tr
import glob
import os


def filter(data):
    r = 4
    theta = np.ones(r)  # vector of parameters of AR model 
    P = np.ones((r, r))
    lambda_f = 0.9  # forrgetting factor
    for n, sample in enumerate(data[r:], r):
        phi = data[n-r:n][::-1]  # r previous samples
        # prediction for sample n, as linear combination of r previous samples
        y_pred_n = np.sum(theta*phi)  # a[0]*y[n-1] + a[1]*y[n-2] + ...

        eps = sample-y_pred_n  # prediction error  

        # New parameters
        k = P @ phi / (lambda_f + tr(phi)@P@phi)
        theta = theta + k*eps
        P = 1/lambda_f * (P - P@phi@tr(phi)*P / (lambda_f + tr(phi)@P@phi))
        # print(k)
        
        

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



