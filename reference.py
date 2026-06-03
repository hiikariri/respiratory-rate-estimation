''' RR_estimator_version date.py

Copyright 2023; copyright holders: Kanchan Kulkarni and Jesse D. Roberts Jr.

MIT LICENSE is applied:

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the "Software"), 
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons 
to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice 
shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, 
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION 
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

'''
# +++++++++++++ Libraries ++++++++++++++++++++++++++++

import os
import sys
import time
import matplotlib.pyplot as plt     # v 3.7.1; matlab plot functions
import numpy as np                  # v 1.24.2; for array, matrices and mathematical operations
import scipy.signal as signal       # v 1.10.1; for filters and signal processing functions

# +++++++++++++ Constants ++++++++++++++++++++++++++++

__FILE_NAME__ = os.path.basename(__file__)
__VERSION__ = '7-21-23'
__AUTHORS__ = 'Kulkarni and Roberts'
__LICENSE__ = 'MIT'

FS = 1000                   # sampling frequency of the input ECG data
F1 = 8 / FS  # 8            # lower bandpass filter frequency
F2 = 40 / FS  # 20          # higher bandpass filter frequency
QRS_INTERVAL = [-40, 40]    # duration of the QRS complex to extract
HEARTBEAT_WINDOW = 16       # number of heartbeats in the moving window to estimate respiratory rate
FFT_LENGTH = 512            # length of power spectrum
FREQ_RANGE = [0.03, 0.3]    # default frequency range to extract respiratory rates
ANALYSIS_CHANNEL = 0        # select which lead to use for RR estimation (0 or 1)
AVERAGING_WINDOW = 16       # define window for extracting median RR

# ecg_exp_parameters is a dictionary that will contain the experimental parameters
ecg_exp_parameters ={
    'file': '',
    'path': '',
    'sample_rate': 0,
    'encoding': 0,
    'low_RR': 0,
    'upper_RR': 0
}

# +++++++++++++ Functions ++++++++++++++++++++++++++++
def greeting():
    """ Provides script information."""
    
    print('Welcome to {}'.format(__FILE_NAME__))
    print('Version: {}'.format(__VERSION__))
    print('Author: {}'.format(__AUTHORS__))
    print('License: {}'.format(__LICENSE__))
    print('Python version: {}\n'.format(sys.version))

    return
    
def get_ecg_exp_parameters():
    """ Collects the user's experimental parameters.
        Uses global ecg_exp_parameters dictionary. """

    _finished = False

    while not _finished:
        print('*** Enter experimental parameters')

        while True:
            # Get data options
            command = input('\nDo you want to test the demo (d) or new (n) ECG data, or quit (q)?: ')
            if command not in ('d', 'n', 'q'):
                print('  try a valid command...')

            # Get the parameters for the demo data
            if command == 'd':
                while True:
                    # Get whether human or sheep demo data will be used
                    demo_data_selection = input('  Do you want to try the human (h) or sheep (s) demo data?: ')
                    if demo_data_selection not in ('h', 's'):
                        print('    try a valid command...')
                    else:
                        break

                # Get the path to demo data
                data_path = input('    Enter path to the demo data: ')
                ecg_exp_parameters['path'] = data_path

                # Assign demo data parameters to ecg_exp_parameters dictionary -
                #  get the RR range of the human and sheep ECG data
                if demo_data_selection == 'h':
                    ecg_exp_parameters['file'] = 'human_1kHz_16bit_demo_7-21-23.bin'
                    ecg_exp_parameters['low_RR'] = 10
                    ecg_exp_parameters['upper_RR'] = 40
                    ecg_exp_parameters['sample_rate'] = 1000
                    ecg_exp_parameters['encoding'] = 'int16'

                else:
                    ecg_exp_parameters['file'] = 'sheep_1kHz_32bit_demo_7-21-23.bin'
                    ecg_exp_parameters['low_RR'] = 3
                    ecg_exp_parameters['upper_RR'] = 40
                    ecg_exp_parameters['sample_rate'] = 1000
                    ecg_exp_parameters['encoding'] = 'int32'

                break

            # Get the parameters to the user's new data
            if command == 'n':
                # Get options for the user data
                user_data_file = input('\nName of data file: ')
                ecg_exp_parameters['file'] = user_data_file
                # Get the path to the user's data
                path = input(' Enter path to the data: ')
                ecg_exp_parameters['path'] = path

                while True:
                    # Get the ECG sampling frequency
                    samp_freq = input(' What is the ECG sampling frequency (Hz): ')
                    if not samp_freq.isdigit():
                        print('  enter numbers...')
                    elif int(samp_freq) < 256:
                        print('\n++++++++++++++++++++++++++++++++++++++++++++++++++')
                        print('+++ To produce good results, the ECG sampling  +++')
                        print('+++ frequency needs to be 256Hz or greater.    +++')
                        print('+++ Please resample your ECG data at a higher  +++')
                        print('+++ rate and then use this script again.       +++')
                        print('++++++++++++++++++++++++++++++++++++++++++++++++++')
                        time.sleep(5)
                        exit_script()
                    else:
                        print('  for script optimization, we will resample your data at 1kHz')
                        ecg_exp_parameters['sample_rate'] = int(samp_freq)
                        break

                while True:
                    # Get the data bit-encoding information
                    int_encoding_options = ['int16', 'int32', 'int64']
                    print(' What is the integer encoding? -')
                    print('  {:^10} ... {:^6}'.format('encoding', 'choice'))
                    for i, option in enumerate(int_encoding_options):
                        print('  {:^10} ... {:^6}'.format(str(option), str(i + 1)))

                    bit_depth_choice = int(input('  >> what is your choice?: '))
                    # print(' you entered: ', bit_depth_choice)

                    if bit_depth_choice not in range(1, len(int_encoding_options) + 1):
                        print('Try a valid choice...\n')
                    else:
                        ecg_exp_parameters['encoding'] = int_encoding_options[bit_depth_choice - 1]
                        break

                while True:
                    # Get the lower RRs
                    lower_RR = input(' Enter anticipated lower RR (bpm): ')
                    if not lower_RR.isdigit():
                        print('  enter numbers...')
                    else:
                        if int(lower_RR) < 3:
                            print('   Caution: this is lower than our validated RR range...')
                        break

                while True:
                    # Get the upper RRs
                    upper_RR = input(' Enter anticipated highest RR (bpm): ')
                    if not upper_RR.isdigit():
                        print('enter numbers...')
                    else:
                        if int(upper_RR) > 40:
                            print('   Caution: this is greater than our validated RR range...')
                        break

                # Assign demo data parameters to ecg_exp_parameters dictionary
                ecg_exp_parameters['low_RR'] = int(lower_RR)
                ecg_exp_parameters['upper_RR'] = int(upper_RR)

                break

            if command == 'q':
                # Quit the script
                print('OK - quiting')
                return ecg_exp_parameters

        print('\nHere is the ECG data parameters you entered-')
        show_ecg_exp_parameters(ecg_exp_parameters)

        while True:
            command = input('  >> do you wish to edit them (y/n)?: ')
            if command not in ('y', 'n'):
                print('    enter valid command...')
            elif command == 'n':
                print('\n')
                return ecg_exp_parameters
            else:
                break

    return ecg_exp_parameters
    
def show_ecg_exp_parameters(_dict):
    """ Show the keys and values in _dict."""

    for _key, _value in _dict.items():
        print("{}{:<15}{}{:>4}".format(' ' * 4, _key, '....', _value))

    return
    
def import_ecgdata(_parameter_dict):
    """ Imports binary data. """

    print('Importing the ECG data...')

    _file = _parameter_dict['file']
    _path = _parameter_dict['path']
    _encoding = _parameter_dict['encoding']

    return np.memmap(filename=_path+_file, dtype=_encoding, mode='r')
    
def ecg_resample(_ecg, _initial_fs, _resampled_f = 1000):
    """ Accepts ecg file that is sampled at _initial_fs Hz rate,
        resamples it at _resampled_f Hz rate, and returns a new ecg file. """

    print('Resampling data at 1000Hz...')
    _scale = _resampled_f / _initial_fs
    _resampled_n = round(len(ecg) * _scale)
    _ecg_resampled = np.interp(np.linspace(0.0, 1.0, _resampled_n, endpoint=False),
                           np.linspace(0.0, 1.0, len(_ecg), endpoint=False),
                           _ecg,  # known data points
                           )

    return _ecg_resampled
    
def detect_rpeaks(_ecg):
    """ Detects R-peaks and corrected R-peaks.
        Required functions: two_average_detector, Rpeak_correction. """

    print('Detecting R-peaks in the ECG data - please wait a few min...')
    # performs Rpeak detection
    _rpeaks = two_average_detector(_ecg)
    print(' --> done with two_average_detector')
    # performs Rpeak correction
    _rpeaks_corr = Rpeak_correction(_ecg, _rpeaks)
    print(' --> done with Rpeak_correction')

    return _rpeaks, _rpeaks_corr
    
def two_average_detector(_unfiltered_ecg):
    """ Detects R-peaks in ECG. It is based on Elgendi, Jonkman, & De Boer (2010).
        Required function: moving_window_ave. """

    _low = F1 * 2
    _high = F2 * 2

    _b, _a = signal.butter(2, [_low, _high], btype='bandpass')

    # applying bandpass filter on raw ECG data
    _filtered_ecg = signal.lfilter(_b, _a, _unfiltered_ecg)

    # 0.12 the Rpeak detection is based on dual moving average window method
    _window1 = int(0.12 * FS)
    # identify qrs interval from first moving average window
    _mwa_qrs = moving_window_ave(abs(_filtered_ecg), _window1)

    _window2 = int(0.6 * FS) #0.6
    # identify duration of the beat based on second moving average window
    _mwa_beat = moving_window_ave(abs(_filtered_ecg), _window2)

    _blocks = np.zeros(len(_unfiltered_ecg))
    _block_height = np.max(_filtered_ecg)

    # identifying segments where the qrs interval magnitude
    #   is larger than the beat magnitude
    for i in range(len(_mwa_qrs)):
        if _mwa_qrs[i] > _mwa_beat[i]:
            _blocks[i] = _block_height
        else:
            _blocks[i] = 0

    _qrs = []

    # Identify R-peaks based on maximum value in each previously identified segment
    for i in range(1, len(_blocks)):
        if _blocks[i - 1] == 0 and _blocks[i] == _block_height:
            start = i

        elif _blocks[i - 1] == _block_height and _blocks[i] == 0:
            end = i - 1

            if end - start > int(0.08 * FS):
                _detection = np.argmax(_filtered_ecg[start:end + 1]) + start
                if _qrs:
                    if _detection - _qrs[-1] > int(0.3 * FS):
                        _qrs.append(_detection)
                else:
                    _qrs.append(_detection)

    return _qrs
    
def Rpeak_correction(signal, rpeaks):
    """ Function that performs Rpeak correction
        - slight shift in actual peaks based on local peaks and valleys. """

    rpeaks = np.array(rpeaks)
    num_peak = rpeaks.shape[0]
    peaks_corrected_list = list()
    for index in range(len(rpeaks)):
        i = rpeaks[index]
        cnt = i
        if cnt-1 < 0:
            break
        if signal[cnt] < signal[cnt-1]:
            while signal[cnt] < signal[cnt-1]:
                cnt -= 1
                if cnt < 0:
                    break
        elif signal[cnt] < signal[cnt+1]:
            while signal[cnt] < signal[cnt+1]:
                cnt += 1
                if cnt < 0:
                    break
        peaks_corrected_list.append(cnt)
    peaks_corrected = np.asarray(peaks_corrected_list)

    return peaks_corrected
    
def moving_window_ave(input_array, window_size):
    """ Calculates the moving window average. """

    moving_window_ave = np.zeros(len(input_array))
    for i in range(len(input_array)):
        if i < window_size:
            section = input_array[0:i]
        else:
            section = input_array[i - window_size:i]

        if i != 0:
            moving_window_ave[i] = np.mean(section)
        else:
            moving_window_ave[i] = input_array[i]

    return moving_window_ave
    
def extract_qrs_complexes(_rpeaks_corr, _ecg):
    """ Determines the R-R interval and then extracts QRS associated with each R-peak. """

    _beatsduration = np.arange(QRS_INTERVAL[0], QRS_INTERVAL[1]+1, 1)
    # applying bandpass filter on the signal to clean the QRS complex
    _b, _a = signal.butter(2, [F1 * 2, F2 * 2], btype='bandpass')
    _data = signal.lfilter(_b, _a, _ecg)

    _beats = np.zeros((len(_beatsduration), len(_rpeaks_corr)))

    # For each heartbeat, extract the QRS complex based on the predefined QRS interval
    #  beats: number of columns equals the number of beats
    #  and the rows correspond to the duration of the QRS complex
    for i in range(len(_rpeaks_corr)):
        dur = np.arange(_rpeaks_corr[i]+QRS_INTERVAL[0], _rpeaks_corr[i]+QRS_INTERVAL[1]+1, 1)
        _beats[:, i] = _data[dur]

    return _beats

def rr_estimator(_dict, _qrses, _rpeaks_corr):
    """ Calculates the RMS value for each QRS complex, calculates the power spectrum,
        and then estimates the RR. """
    
    _resp_range = [_dict['low_RR'], _dict['upper_RR']]
    _rms = np.zeros(len(_qrses[0]))

    # Calculate the RMS
    for i in range(len(_rms)):
        _rms[i] = np.sqrt(np.mean(_qrses[:, i]**2))

    # Generate power spectrum of the RMS values and estimate RR
    #  _freq_vector: frequency vector used for mapping (0-0.5Hz)
    #  rrint: RR-intervals
    _freq_vector = np.linspace(0, 0.5, FFT_LENGTH//2+1)
    _rrint = np.diff(_rpeaks_corr)
    _resprate = []

    for i in range(len(_rms)):
        if i < HEARTBEAT_WINDOW:
            _section = _rms[0:i]
            _rrint_med = _rrint[0:i]
        else:
            _section = _rms[i - HEARTBEAT_WINDOW:i]
            _rrint_med = _rrint[i - HEARTBEAT_WINDOW:i]
            _rrint_med = np.median(_rrint_med)

            if _rrint_med > 100:
                # Calculate frequency mapping range based on median HR and predefined RR range
                _fpeaks = np.where((_freq_vector > _resp_range[0] * _rrint_med / 60000) & (_freq_vector < _resp_range[1] * _rrint_med / 60000))
                _fpeaks_arr = np.asarray(_fpeaks[0])
            else:
                # Use default frequency map in case of erroneous HR/RR0intervals
                _fpeaks = np.where((_freq_vector > FREQ_RANGE[0]) & (_freq_vector < FREQ_RANGE[1]))
                _fpeaks_arr = np.asarray(_fpeaks[0])

            # power spectrum of moving window of RMS values
            _spectrum = np.fft.fft(_section-np.mean(_section), FFT_LENGTH)
            rf = _spectrum[0:FFT_LENGTH//2+1]
            _spectrum_mod = rf*np.conj(rf)
            # find location of power spectrum peak
            _respmax = np.argmax(_spectrum_mod[_fpeaks])+_fpeaks_arr[0]
            # estimate respiratory rate
            _resprate.append(_freq_vector[_respmax] * 60000/_rrint_med)

    return _resprate
    
def smooth_RR(_dict, _resprate):
    """ Smooths RR by determining the median using a moving window preceding RRs.
        It also saves the smoothed estimated RRs to a file in CSV format. """

    # Make array from estimated RR values
    _resprate = np.asarray(_resprate)
    # Make array to hold index number and smoothed values
    _index_resprate_mwa = np.zeros(len(_resprate))
    _resprate_mwa = np.zeros(len(_resprate))
    # Get path to where smoothed data will be saved
    _path = _dict['path']

    # Smooth the data by determining the median value in a window
    for i in range(len(_resprate)):
        _index_resprate_mwa[i] = i

        if (i > 0) & (i < AVERAGING_WINDOW):
            _resprate_mwa[i] = _resprate[i]
        elif i > AVERAGING_WINDOW:
            _ind = np.arange(i-AVERAGING_WINDOW, i, 1)
            # median RR for each moving window
            _resprate_mwa[i] = np.median(_resprate[_ind])
        else:
            _resprate_mwa[i] = _resprate[i]

    # Round smoothed estimated RR's
    _resprate_mwa = np.around(_resprate_mwa, decimals=2, out=None)

    # Combine index and smoothed RR values
    _indexed_resprate_mwa = np.column_stack((_index_resprate_mwa,_resprate_mwa))

    # Save the estimated RR's to a file
    print('Now saving respiratory rate data...')
    #np.savetxt(_path+"resprate.csv", _resprate_mwa, delimiter=", ", fmt='%f')
    np.savetxt(_path + "resprate.csv", _indexed_resprate_mwa, delimiter=", ", fmt='%f', header = 'sample, estRR', comments='')

    return _resprate_mwa
    
def show_results(_ecg, _rpeaks_corr, _smooth_resprate):
    """ Constructs R-peak and estimated RR plots. """

    print('Now constructing plots of results...')

    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.suptitle('Results', fontsize=16)

    ax1.set_title('R-peaks')
    ax1.set_xlabel('data point')
    # Plot ecg data as series
    ax1.plot(ecg, label='ECG')
    # Overlay plot of corrected R-peaks
    ax1.plot(_rpeaks_corr, _ecg[rpeaks_corr], '*', label='R-peaks')
    ax1.legend(loc='upper left')

    ax2.set_title('Smoothed estimated RR')
    ax2.set_xlabel('ecg cycle')
    ax2.set_ylabel('breaths per min')
    ax2.plot(_smooth_resprate)
    plt.show()

    return
    
def exit_script():
    ''' Exits the script. '''

    print('\nThank you for using {}'. format(__FILE_NAME__))
    print('\n*** Exiting Python ***')
    quit()

# +++++++++++++ Main ++++++++++++++++++++++++++++++++++

greeting()

# Get the ecg data parameters
ecg_exp_parameters = get_ecg_exp_parameters()

# Get the ecg data
ecg = import_ecgdata(ecg_exp_parameters)
print(' --> done importing data')

# Resample the ecg data if not sampled at 1kHz
if ecg_exp_parameters['sample_rate'] != 1000:
    ecg = ecg_resample(ecg, _initial_fs=ecg_exp_parameters['sample_rate'])
    print(' --> done resampling data')

# Detect the R-peaks
rpeaks, rpeaks_corr = detect_rpeaks(ecg)
print(' --> done detecting R-peaks')

# Extract the QRSes
qrses = extract_qrs_complexes(rpeaks_corr, ecg)
print(' --> done extracting QRSes')

# Estimate the RR
resprate = rr_estimator(ecg_exp_parameters, qrses, rpeaks_corr)
print(' --> done estimating the RRs')

# Smooth and save the estimated RRs
smooth_resprate = smooth_RR(ecg_exp_parameters, resprate)
print(' --> done smoothing the estimated RR data')

# Plot the results
show_results(ecg, rpeaks_corr, smooth_resprate)
print(' --> done graphing the data')

exit_script()