from concurrent.futures import thread
from numpy.core.fromnumeric import size
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
    @return: smoothed_data - without impulse noise
             theta_arr - r x len(data) array of parameters
             std_dev_array - array of standard deviation
    """
    theta = np.ones(r)  # vector of parameters of AR model 
    P = np.eye(r, dtype=np.float64)
    lambda_f = 0.95  # forgetting factor
    smoothed_data = np.copy(data)
    M = 20
    last_errors = np.ones(M)*np.max(np.abs(data))  # array of last M errors - to calculate st deviation
    std_dev = np.sqrt(np.sum(np.square(last_errors))/M)  # init standard deviation - very high
    corr_in_row = 0  # number of detected noise samples in a row

    std_dev_arr = np.zeros(size(data))
    theta_arr = np.ones((r, size(data)))

    for n, sample in enumerate(data[r:-1], r):
        phi = smoothed_data[n-r:n][::-1]  # r previous samples
        # prediction for sample n, as linear combination of r previous samples
        y_pred_n = np.sum(theta*phi)  # a[0]*y[n-1] + a[1]*y[n-2] + ...

        eps = sample-y_pred_n  # prediction error 
        # Update standard deviation 
        last_errors[:-1] = last_errors[1:]
        last_errors[-1] = eps  # shifting errors
        std_dev = np.sqrt(np.sum(np.square(last_errors))/M)
        if np.abs(eps) < 3*std_dev: 
            # Good sample
            # Update parameters
            k = P @ phi / (lambda_f + tr(phi)@P@phi)
            theta = theta + k*eps  # New AR parameters
            P = 1/lambda_f * (P - P@phi@tr(phi)*P / (lambda_f + tr(phi)@P@phi))
            corr_in_row = 0
        else:
            # Noise sample
            smoothed_data[n] = 0.5*(data[n-1]+data[n+1])
            corr_in_row += 1
            
        theta_arr[:, n] = theta
        std_dev_arr[n] = std_dev

    return smoothed_data, theta_arr, std_dev_arr

for filepath in glob.iglob('wav/*.wav'):
    # Read audio data
    samplerate, data = wavfile.read(filepath)
    r = 4
    new_data, parameters, std_dev_arr = filter(data, r)
    
    # Write data without impulse noise to file
    basename = os.path.basename(filepath)
    new_filepath = os.path.join("smoothed", os.path.splitext(basename)[0] + "_smoothed.wav")
    wavfile.write(new_filepath, samplerate, new_data)
    print(new_filepath)

    # Plot original and smoothed data, save img
    plt.figure(0)
    plt.plot(data)
    plt.plot(new_data)
    plt.legend(["original", "smoothed"])
    plt.title("Audio: {}".format(basename))
    plt.savefig('img\\img_{}.png'.format(basename))

    plt.figure(1)
    legend_par = []
    for i in range(r):
        plt.plot(parameters[i, :])
        legend_par.append(f"a_{i}")
    plt.legend(legend_par)
    plt.title("Paramters - Audio: {}".format(basename))
    plt.savefig('img\\param_{}.png'.format(basename))

    plt.figure(2)
    plt.plot(std_dev_arr[25:])  # cut out first high values
    plt.title("Standard Deviation - Audio: {}".format(basename))
    plt.savefig('img\\std_dev_{}.png'.format(basename))
    plt.show()
