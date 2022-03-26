from scipy.io import wavfile
import numpy as np
from numpy import transpose as tr
import glob
import os
from matplotlib import pyplot as plt


def filter(data, r):
    r = 4  # order of AR model
    theta = np.ones(r)  # vector of parameters of AR model 
    P = np.eye(r, dtype=np.float64)
    lambda_f = 0.95  # forgetting factor
    smoothed_data = np.copy(data)
    M = 20
    last_errors = np.ones(M)*np.max(np.abs(data))  # array of last M errors - to calculate st deviation
    std_dev = np.sqrt(np.sum(np.square(last_errors))/M)
    corr_in_row = 0  # number of detected noise samples in a row

    for n, sample in enumerate(data[r:-1], r):
        phi = smoothed_data[n-r:n][::-1]  # r previous samples
        # prediction for sample n, as linear combination of r previous samples
        y_pred_n = np.sum(theta*phi)  # a[0]*y[n-1] + a[1]*y[n-2] + ...

        eps = sample-y_pred_n  # prediction error 

        if np.abs(eps) < 3*std_dev: 
            # Good sample
            last_errors[:-1] = last_errors[1:]
            last_errors[-1] = eps  # shifting errors
            std_dev = np.sqrt(np.sum(np.square(last_errors))/M)

            k = P @ phi / (lambda_f + tr(phi)@P@phi)
            theta = theta + k*eps  # New parameters
            P = 1/lambda_f * (P - P@phi@tr(phi)*P / (lambda_f + tr(phi)@P@phi))
            corr_in_row = 0
        else:
            # Noise sample
            smoothed_data[n] = 0.5*(data[n-1]+data[n+1])
            corr_in_row += 1

    return smoothed_data


for filepath in glob.iglob('wav/*.wav'):
    # Read audio data
    samplerate, data = wavfile.read(filepath)

    if filepath in []: # debug temp - choose files to process  'wav\\01.wav', 'wav\\02.wav', 'wav\\03.wav', 'wav\\04.wav' ,'wav\\05.wav', 
        print(f"skipp: {filepath}")
        continue
    new_data = filter(data, 4)

    # Write data without impulse noise to file
    basename = os.path.basename(filepath)
    new_filepath = os.path.join("smoothed", os.path.splitext(basename)[0] + "_smoothed.wav")
    wavfile.write(new_filepath, samplerate, new_data)
    print(new_filepath)

    plt.subplot(2, 2, 1)
    plt.plot(data)
    plt.title("original")
    plt.subplot(2, 2, 2)
    plt.plot(new_data)
    plt.title("smoothed")

    plt.subplot(2, 2, 3)
    plt.plot(data)
    plt.plot(new_data)
    plt.legend(["original", "smoothed"])

    plt.subplot(2, 2, 4)
    plt.plot(new_data)
    plt.plot(data)
    plt.legend(["smoothed", "original"])
    plt.show()



