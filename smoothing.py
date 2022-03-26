from scipy.io import wavfile
import numpy as np
from numpy import transpose as tr
import glob
import os
from matplotlib import pyplot as plt


def filter(data, r=4):
    """
    Remove impulse noise from audio data using autoregressive model
    @param data: original audio data
    @param r: order of autoregressive model
    @return: smoothed data - without impulse noise
    """
    r = 4  # order of AR model
    theta = np.ones(r)  # vector of parameters of AR model 
    P = np.eye(r, dtype=np.float64)
    lambda_f = 0.95  # forgetting factor
    smoothed_data = np.copy(data)
    M = 20
    last_errors = np.ones(M)*np.max(np.abs(data))  # array of last M errors - to calculate st deviation
    std_dev = np.sqrt(np.sum(np.square(last_errors))/M)  # init standard deviation - very high
    corr_in_row = 0  # number of detected noise samples in a row

    for n, sample in enumerate(data[r:-1], r):
        phi = smoothed_data[n-r:n][::-1]  # r previous samples
        # prediction for sample n, as linear combination of r previous samples
        y_pred_n = np.sum(theta*phi)  # a[0]*y[n-1] + a[1]*y[n-2] + ...

        eps = sample-y_pred_n  # prediction error 

        if np.abs(eps) < 3*std_dev: 
            # Good sample
            # Update standard deviation 
            last_errors[:-1] = last_errors[1:]
            last_errors[-1] = eps  # shifting errors
            std_dev = np.sqrt(np.sum(np.square(last_errors))/M)
            # Update parameters
            k = P @ phi / (lambda_f + tr(phi)@P@phi)
            theta = theta + k*eps  # New AR parameters
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
    new_data = filter(data, 4)

    # Write data without impulse noise to file
    basename = os.path.basename(filepath)
    new_filepath = os.path.join("smoothed", os.path.splitext(basename)[0] + "_smoothed.wav")
    wavfile.write(new_filepath, samplerate, new_data)
    print(new_filepath)

    # Plot original and smoothed data, save img
    plt.plot(data)
    plt.plot(new_data)
    plt.legend(["original", "smoothed"])
    plt.title("Audio: {}".format(basename))
    plt.savefig('img\\img_{}.png'.format(basename))
    plt.show()



