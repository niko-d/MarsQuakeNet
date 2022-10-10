"""
Methods for MarsQuakeNet

Developed for MarsQuakeNet / InSight data.
Author: N.Dahmen - ETH Zurich

"""
import glob
import numpy as np
import numpy.ma as ma
import scipy
from scipy.signal import savgol_filter, find_peaks
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, FormatStrFormatter
from matplotlib import gridspec

from obspy import read, Stream, UTCDateTime
from obspy.core.event import Event, Origin, Arrival, Pick, EventDescription , CreationInfo, Comment

from utils import sol_span_in_utc, utc2lmst
from sklearn.preprocessing import RobustScaler
from datetime import datetime

from obspy.signal.trigger import classic_sta_lta
import sys
import glob
from natsort import natsorted

# plotting
plt.rcParams['figure.figsize'] = [16, 12]
plt.rcParams['font.size'] = 12
# stft param
kwargs = {'nperseg': 256,
          'fs': 20,
          'nfft': 510,
          'boundary': 'zeros'}

# Define reference times for utc to lmst/sol conversion
REFERENCE_TIME_INSIGHT_LANDING = UTCDateTime("2018-11-26T05:10:50.336037Z")
SOL01_START_TIME = UTCDateTime("2018-11-27T05:50:25.580014Z")
SOL02_START_TIME = UTCDateTime("2018-11-28T06:30:00.823990Z")
SECONDS_PER_MARS_DAY = SOL02_START_TIME - SOL01_START_TIME - 0.000005
SECONDS_PER_EARTH_DAY = 86400.


def sol_span_in_utc(sol, sol0_start_utc=REFERENCE_TIME_INSIGHT_LANDING):
    """
    Developed for InSight mission to Mars.
    Author: Savas Ceylan - ETH Zurich
            Martin van Driel - ETH Zurich`
    Returns start and end times in UTC for a given sol.
    """
    utc_representation = \
        UTCDateTime(sol * SECONDS_PER_MARS_DAY) + float(sol0_start_utc)

    return utc_representation, utc_representation + SECONDS_PER_MARS_DAY

def utc2lmst(utc_time, sol0_start_utc=REFERENCE_TIME_INSIGHT_LANDING,
             sol_dtype='int'):
    """
    Developed for InSight mission to Mars.
    Author: Savas Ceylan - ETH Zurich
            Martin van Driel - ETH Zurich
    Convert UTC to LMST. Default sol-0 time is InSight landing time
    in UTC. Returned LMST counts from Linux epoch; date value showing
    the sol number.
    Return value is a tuple of LMST as UTCDateTime instance and sol number.
    Sol number can be integer or float. If float, it includes decimal
    fractions of sol as well.
    """
    # Cast to UTCDateTime, if datetime is given. This is useful to avoid
    # long type casting statements while plotting
    if isinstance(utc_time, datetime):
        utc_time = UTCDateTime(utc_time)

    _elapsed_mars_days = (utc_time - sol0_start_utc) / SECONDS_PER_MARS_DAY

    _time = UTCDateTime((_elapsed_mars_days - 1) * SECONDS_PER_EARTH_DAY)

    # Return a tuple with local Mars time as UTCDateTime and sol number
    if sol_dtype == 'float' or sol_dtype is None:
        return _time, _elapsed_mars_days
    else:
        return _time, np.int(np.floor(_elapsed_mars_days))


def seis_by_sol(sol, inv, channel, remove='sensitivity', pre_filt=(1 / 50, 1 / 30, 9, 9.3),
                output='VEL', return_gaps=False, mask=True, overlap=900, check4data=False, keep_overlap=False):
    """
    Get seismic data by sol:
    local mseed-files are selected and loaded (saved in utc year folder structure), for standard VBB and SP channels
    Stream is trimmed, merged (gaps are saved), detrended, sens/resp (+ pre filter) removed, rotated, gaps cutout,
    masked (optional), cut to include up to 3h of overlap
    input (required): sol, inv file, channel (02.BH - 03.BH - 58.BZ - 67.SH - 68.SH), remove='sensitivity'/'response'
    input (optional): prefilter used when remove='response', overlap [s],
    bool return_gaps, bool mask gaps, bool check4data (checks for local preprocessed VBB data, much faster)
    bool keep_overlap (relict) - sets overlap=900[s]
    return: preprocessed stream and gaps (optional)
    requires: local mseed data, folders sorted by years, sol_span_in_utc(sol), obspy
    """
    preprocessed = False
    if check4data:  # search for local preprocessed data
        # select folder for 10 or 20sps VBB data
        if channel == '10sps':
            path = "./Data/Waveforms/02.BH/PreProcessed/SEIS/"
            file_preprocessed = glob.glob(path + "*00.BHZNE.Sol" + str(sol).zfill(4) + ".mseed")
        else:  # 20 sps file
            path = "./Data/Waveforms/02.BH/PreProcessed/SEIS/"
            file_preprocessed = glob.glob(path + "*02.BHZNE.Sol" + str(sol).zfill(4) + ".mseed")

        # if file found, read stream and load gaps (in local folder)
        if len(file_preprocessed) > 0:
            st = read(file_preprocessed[0])
            try:
                if channel == '10sps':
                    gap_path = "./Data/Waveforms/02.BH/PreProcessed/Gaps/gaps_sol" + str(sol) + '.csv'
                else:  # 20 sps file
                    gap_path = "./Data/Waveforms/02.BH/PreProcessed/Gaps/gaps_sol" + str(sol) + '.csv'

                gaps = np.loadtxt(gap_path, delimiter=',', dtype='str',
                                  converters={4: UTCDateTime, 5: UTCDateTime})
                gaps = gaps.tolist()
            except:  # if no gap file available
                print('No gaps')
                gaps = []
            for j in range(len(gaps)):  # convert to UTCDateTime file
                gaps[j][4] = UTCDateTime(gaps[j][4])
                gaps[j][5] = UTCDateTime(gaps[j][5])
            preprocessed = True
        else:
            print('No preprocessed data found - preprocessing...')
    else:
        if channel == '10sps':
            print('10 sps data not preprocessed')
            return
    # get utc of sol
    utc = sol_span_in_utc(sol)[0]
    utc_end = sol_span_in_utc(sol)[1]

    # start of preprocessing block
    # get day of year to identify mseed file
    if not preprocessed:
        if inv==None:
            print('Provide inventory file (inv) or preprocessed data')
            print('https://www.seis-insight.eu/en/science/seis-data/seis-metadata-access')

        doy_start = (utc-3600*3).julday  # min. 3 hour buffer
        doy_end = (utc_end+3600*3).julday  # min. 3 hour buffer
        if np.abs(doy_end - doy_start) > 300:  # new year
            doy_end += 365
        year = utc.year

        print('Sol: ', sol, ' DOY: ', doy_start, ' Year: ', year)

        # load files
        st = Stream()
        files = []

        # load mseed file of corresponding doy
        doys = range(doy_start, doy_end+1)

        for doy in doys:
            if doy<366: # if in same year
                files += glob.glob("/home/niko/Schreibtisch/Data/" + channel + "/" + str(year) + "/*" + channel[-2:]
                                + "*" + str(doy).zfill(3))
            else:  # if in next year
                doy-=365
                year=utc.year+1
                files += glob.glob("/home/niko/Schreibtisch/Data/" + channel + "/" + str(year) + "/*" + channel[-2:]
                                + "*" + str(doy).zfill(3))

        if len(files) == 0:
            return

        for f in files:  # load in stream
            st += read(f)
        # stream preprocessing
        try:
            # trim data to sol +-3h
            st.trim(utc - 3 * 3600, utc_end + 3 * 3600)  # cut sol-long stream with buffer in beginning/end
            gaps = st.get_gaps()  # get gaps of data

            st.merge(method=1, fill_value=0)  # Fill gaps with 0 for preprocessing
            st.detrend("constant")

            # optional: remove either sensitivity or response
            if remove == 'sensitivity':
                st.remove_sensitivity(inventory=inv)
                print('Gain removed')
            if remove == 'response':
                sampling = st[0].stats.sampling_rate
                st.remove_response(inventory=inv, output=output, pre_filt=pre_filt)
                print('Instrument response corrected')
            # rotate stream to ZNE
            st.rotate(method="->ZNE", inventory=inv, components=["UVW"])

            # get zeroes back into gaps so that gap detector can work with it
            # by cuting out gaps with saved gap list, and filling it with zeros
            if len(gaps) > 0:
                for i in range(len(gaps)):
                    st.cutout(gaps[i][4], gaps[i][5])  # cut out where gaps are
                st.merge(method=0, fill_value=0)  # merge back and fill gaps with zeroes
        except:
            print('Failed - check preprocessing')
            return
        # end of preprocessing block

    # cut to overlap
    if keep_overlap: # relict
        overlap = 900
    if overlap == 0:  # changed, remove first if
        st.trim(utc, utc_end, pad=True, fill_value=None)  # trim to actual sol-long stream
    else:
        st.trim(utc - overlap, utc_end + overlap, pad=True, fill_value=None)  # trim to actual sol-long stream

    if mask:  # optional: mask zeros=gaps
        # Mask filled 0s
        st[0].data = ma.masked_where(st[0].data == 0, st[0].data)
        st[1].data = ma.masked_where(st[1].data == 0, st[1].data)
        st[2].data = ma.masked_where(st[2].data == 0, st[2].data)
        # if not mask:
        #     st = st.split()

    if len(st) == 0:
        print('No data or wrong channel (02.BH - 03.BH - 58.BZ - 67.SH - 68.SH)')
    # return stream (+ gaps)
    if return_gaps:
        return st, gaps
    else:
        return st

#  other functions

def get_event_start(row):
    # consumes rows from detection_list
    """
    Get events start in utc from detection list
    (computed relative to peak of detection)
    input: row of detections list
    return: start and end in utc
    """
    utc_peak = UTCDateTime(row[1])  # for loaded list tin str format
    start_utc = utc_peak - np.abs(row[7] - row[5]) * 6.36
    end_utc = utc_peak + np.abs(row[7] - row[6]) * 6.36
    return start_utc, end_utc

def find_freq(f, f_val):
    """
    Finds index of freq. value (first freq. index > freq. value)
    input: freq vector, freq. value
    return: index
    """
    for i in range(len(f)):
        if f[i] > f_val:
            return i


# Class to get data and make predictions, sol-wise
class predict_masks:
    """
    Class to get data, compute stft, normalise and make predictions
    Attributes: clip_val=50 (do not change)
    Methods:
    get_stft_sol: get seismic data for full sol, cut into 27min windows and compute stft
    normalize_batch_perc: normalise stft data
    make_predictions: make prediction based on norm. stft
    norm_output: normalise predicted output so that event+noise=1
    """
    def __init__(self):
        self.clip_val = 50
        # self.norm_separately = True

    def get_stft_sol(self,sol,inv,stft_parameters={'nperseg': 256, 'fs': 20, 'nfft': 510, 'boundary': 'zeros'}):
        """
        Get seismic data for full sol, cut into 27min windows and compute stft
        input: sol, inv file (station xml)
        output: adds to instance - sol, array with stft data, same stft normalised, stream gaps,
        list with start&end of time windows
        """
        self.sol = sol
        sol_data = np.zeros((110, 256, 256, 6))  # init. numpy array for full-sol data
        sol_data_norm = np.zeros((110, 256, 256, 6))  # init. numpy array for full-sol data, nomrlaised
        utc_list = []  # list with utc times of each sample (27min window)
        # get seismic data
        # st, gaps = seis_by_sol(sol=sol, inv=inv, channel='02.BH', remove='response', return_gaps=True,
        #                        output='VEL', mask=True, keep_overlap=True)
        st, gaps = seis_by_sol(sol=sol, inv=inv, channel='02.BH', remove='response', return_gaps=True,
                               output='VEL', mask=True, keep_overlap=True, check4data=True)

        stream_start = st[0].stats.starttime + 1 * 6.4  # set starttime for sol-data, includes 900s overlap

        for i in range(110):  # loop over 110 samples (27min windows) with 50% overlap + overlap to previous/next sol
            # start and end time of sample
            t_start = stream_start + i / 2 * 1628 - 1*6.4 # sample length=1628s
            t_end = stream_start + (i / 2 + 1) * 1628  + 1*6.4 # 6.4s added in beginning and end, removed later
            # copy stream and get stream matching sample time window
            st_noise = st.copy()
            st_noise.trim(t_start, t_end)

            for j in range(3):  # loop through ZNE components
                comp = ['Z','N','E'][j]   # get component
                data = st_noise.select(component=comp)[0].data  # get data from stream for component
                f, t, Zxx_noise = scipy.signal.stft(data, **stft_parameters)  # compute stft
                Zxx_noise = Zxx_noise[:,1:-1]  # remove first/last bin
                sol_data[i, :, :, j*2] = Zxx_noise.real  # get real part of stft
                sol_data[i, :, :, j*2+1] = Zxx_noise.imag  # get imag part of stft

            utc_list.append([t_start, t_end])  # append list time widnows to utc list

        # normalise data, separately for each component
        sol_data_norm[:,:,:,0:2] = self.normalize_batch_perc(sol_data[:,:,:,0:2], limit=self.clip_val)  # Z
        sol_data_norm[:,:,:,2:4] = self.normalize_batch_perc(sol_data[:,:,:,2:4], limit=self.clip_val)  # N
        sol_data_norm[:,:,:,4:6] = self.normalize_batch_perc(sol_data[:,:,:,4:6], limit=self.clip_val)  # E
        # add data as attributes
        self.sol_data_norm = sol_data_norm
        self.sol_data = sol_data
        self.gaps = gaps
        self.utc_list = utc_list

    def normalize_batch_perc(self,data, quantile_range=(25, 75), unit_variance=False, limit=50, clipping='constant'):
        """
        Normalizes batches sample-wise by removing mean / dividing by std
        Separately for each sample and component and real/imag
        Input: stft data, optional values
        Output: normalised data
        """
        batch_size = data.shape[0]  # get number of batches in data array
        data_norm = np.zeros(data.shape)  # init. normalised array

        for i in range(batch_size):  # loop through batch size
            data_real = data[i, :, :, 0]  # get real part
            data_imag = data[i, :, :, 1]  # imag part
            # Define sklearn robust scaler and apply
            scaler = RobustScaler(unit_variance=unit_variance,quantile_range=quantile_range)
            data_real_norm = scaler.fit_transform(np.expand_dims(data_real.flatten(), 1)).reshape((256, 256))
            data_imag_norm = scaler.fit_transform(np.expand_dims(data_imag.flatten(), 1)).reshape((256, 256))

            # clip high values
            stretch = limit / 200  # strecch variable only used for clipping !=constant
            if clipping == 'constant':  # hard clip of values > limit
                data_real_norm[data_real_norm > limit] = limit
                data_real_norm[data_real_norm < -limit] = -limit
                data_imag_norm[data_imag_norm > limit] = limit
                data_imag_norm[data_imag_norm < -limit] = -limit
            elif clipping == 'tanh':  # soft clipping using tanh
                data_real_norm = np.tanh(stretch * data_real_norm)
                data_imag_norm = np.tanh(stretch * data_imag_norm)
            elif clipping == 'sigmoid':  # soft clipping using sigmoid
                data_real_norm = (1 / (1 + np.exp(-stretch * data_real_norm))) * 2 - 1
                data_imag_norm = (1 / (1 + np.exp(-stretch * data_imag_norm))) * 2 - 1
            elif clipping == 'arctan':  # soft clipping using arctan
                data_real_norm = 2 / np.pi * np.arctan(stretch * data_real_norm)
                data_imag_norm = 2 / np.pi * np.arctan(stretch * data_imag_norm)
            else:  # no clipping
                data_real_norm = data_real_norm
                data_imag_norm = data_imag_norm
            # add normalised data to batch array
            data_norm[i, :, :, 0] = data_real_norm
            data_norm[i, :, :, 1] = data_imag_norm
        return data_norm

    def make_predictions(self,model_even,model_odd):
        """
        Make prediction with model based on norm. input
        Separately for each sample and component and real/imag
        Input: models
        Output: adds to instance - predicted masks (normalised) and used model
        Requires: norm_output - normalised output so that event and noise add up to 1=100%
        """
        # Make prediction - using 'even-sol model' for odd-sols and v.v.
        # select model
        if self.sol%2==1:  # odd sol, use model trained on even-sols
            y_window = model_even.predict(self.sol_data_norm)
            model_sol = model_even.name
        else:  # even sol, use model trained on odd-sols
            y_window = model_odd.predict(self.sol_data_norm)
            model_sol = model_odd.name

        # normalise output and add to instance
        self.y_window = self.norm_output(y_window)
        self.model_sol = model_sol

    def norm_output(self,mini_batch):
        """
        Normalise event and noise mask (so they add up to 1=100%)
        Input: ZNE data
        Output: normalised ZNE data
        """
        for sample in mini_batch:
            # Z
            sum_sample = sample[:,:,0]+sample[:,:,1]
            sample[:,:,0]/=sum_sample
            sample[:,:,1]/=sum_sample
            # N
            sum_sample = sample[:,:,2]+sample[:,:,3]
            sample[:,:,2]/=sum_sample
            sample[:,:,3]/=sum_sample
            # E
            sum_sample = sample[:,:,4]+sample[:,:,5]
            sample[:,:,4]/=sum_sample
            sample[:,:,5]/=sum_sample
        return mini_batch

# class to make predictions
class check_prediction:
    """
    Use prediction to make sol-wise plot, extract detections with amplitudes, event family, etc
    Attributes: attributes produced by predict_masks class
    Requires: attributes produced by predict_masks (make_predictions)
    Methods:
    plot_detecions: main method to produce sol-wise masks plots and obtain detections
    get_events_sol: get catalogued events / sus signals on specific sol
    get_landeractivity_sol: get lander activity on specific sol
    sort_gaps: sort gaps, remove doubles and short gaps
    find_grid: helper function for plotting
    check_label: helper function to get event type label and quality
    get_gaps_box: helper function to convert between utc and time bins in timewindows
    get_detections: extract detections from integrated mask timeseries
    check_detect: compare detections against mqs events, sus signals etc
    check_significance: compute detection values, get significant detections
    find_start_end: helper function
    remove_gap_detections: removes detections coinciding with gaps
    num2boxindex: helper function
    classify_detections: classification of detections in event family or type
    get_ampl: amplitude computation
    get_ev_vals: extract median amplitude values in freq. bands
    classify_median_vals: helper function to classify median values in event types
    get_ev_type_color: helper function
    """
    def __init__(self,sol_object):
        self.y_window = sol_object.y_window
        self.sol = sol_object.sol
        self.utc_list = sol_object.utc_list
        self.gaps = sol_object.gaps
        self.sol_data = sol_object.sol_data
        self.threshold = 100 # min. threshold to collect as detections
        self.lf_threshold = 319  # lf detection specific threshold, used to mark events in plot which are above
        self.hf_threshold = 419 # hf detection specific threshold, used to mark events in plot which are above
        self.percentile_val = 90 # percentile value for peak amplitude computation

    def plot_detections(self, list_ev=[], list_susp=[], list_lander=[], save_dir=None,
                       plot_mask=1, skip_repeated=False,check_detections=True, plot_marker=True, classify=True, plot_legend=False):
        """
        Make sol-wise plot, extract detections with amplitudes, event family, etc.
        Input: attributes produced by predict_masks class,
        (optional) lists of event, sus signals, lander activity,
        save_dir: plots are saved if defined, otherwise shown
        plot_mask: 0-show input, 1-show event masks, 2-show extracted events, 3-show extracted noise
        boolean values: skip_repeated-to skip every second row (with 50% overlap), check_detections-check detections,
        plot_marker-plot event marker, classify-classify detections,plot_legend-plot a legend
        Output: sol-wise plot, detection lists
        Requires: get_events_sol, get_landeractivity_sol, sort_gaps, find_grid, check_label, get_gaps_box,
        get_detections, check_detect, check_significance, find_start_end, remove_gap_detections, num2boxindex:
        classify_detections, get_ampl, get_ev_vals, classify_median_vals, get_ev_type_color:
        """
        # separate list of events, sus signals and lander activity in two lists with start and end (last two can be [])
        event_list_start, event_list_end = self.get_events_sol(list_ev, return_name=True, include_overlap=True)
        susp_list_start, susp_list_end = self.get_events_sol(list_susp)
        lander_list_temp = self.get_landeractivity_sol(list_lander, include_overlap=False)

        # start figure specs
        fig = plt.figure(figsize=[12, 8])
        num_samples = self.y_window.shape[0]  # number of windows
        dim = int(np.ceil(np.sqrt(num_samples)))  # dimension for plot with dimxdim subplots
        spec = gridspec.GridSpec(ncols=10 * dim + 5, nrows=16, figure=fig,
                                 height_ratios=[9, 9, 1, 9, 9, 1, 9, 9, 1, 9, 9, 1, 9, 9, 1, 1])  # spec
        spec.update(wspace=0.0, hspace=0.0)  # spacing
        # end figure specs

        # list to colletc detection values, others
        all_band_sol1 = []  # init. list for extracted sol values from row 0,2,4,...
        all_band_sol2 = []  # same for row 1,3,5,...
        gap_windows = []  # list for gap windows

        gaps_sorted = self.sort_gaps()  #  sort gaps, skip short gaps, remove doubles

        # main loop goinf through 110 samples = 27min time windows
        for i in range(num_samples):  # loop over 27min windows
            if skip_repeated and i % 2 != 0:  # optional. skip every second repeated row
                continue;

            mask = self.y_window[i, :, :, :]  # get predicted masks

            # prepare required data for plotting
            if plot_mask == 2:  # extract event if required
                noisy_signal = self.sol_data[i, :, :, 0] + 1j * self.sol_data[i, :, :, 1]
                extracted_event = noisy_signal * mask[:, :, 0]
            if plot_mask == 0:  # extract event if required
                noisy_signal = self.sol_data[i, :, :, 0] + 1j * self.sol_data[i, :, :, 1]
                extracted_event = noisy_signal
            if plot_mask == 3:  # extract event if required
                noisy_signal = self.sol_data[i, :, :, 0] + 1j * self.sol_data[i, :, :, 1]
                extracted_event = noisy_signal * mask[:, :, 1]

            # extract data from event masks to check for new events, get detections
            if check_detections:
                mask_temp = np.copy(mask[:, :, 0])#+ mask[:, :, 2] + mask[:, :, 4]
                mask_temp[mask_temp < 0.25]=0  # ignore small mask values

                mask_temp[:,0:5]=0  #  ignore 5 columns of mask values on left edge of window
                mask_temp[:,251:256]=0  #  ignore  5 columns of mask values on right edge of window

                all_bool = np.sum(mask_temp, axis=0)  # sum along freq. axis
                if i % 2 == 0:  # append on correct list, two lists for overlapping windows
                    all_band_sol1.append(all_bool)
                else:
                    all_band_sol2.append(all_bool)

            # prepare plotting
            # find row, col for each index
            row_grid, col_grid = self.find_grid(i)
            if i % 2 == 0:  # every second row: add subplots, make frame blue
                f_ax = fig.add_subplot(spec[row_grid, col_grid:col_grid + 10])
                f_ax.spines['left'].set_color('tab:blue')
                f_ax.spines['right'].set_color('tab:blue')
                f_ax.spines['top'].set_color('tab:blue')
            else:  # every second row with indent add subplots, make frame red
                f_ax = fig.add_subplot(spec[row_grid, col_grid + 5:col_grid + 15])
                f_ax.spines['left'].set_color('tab:red')
                f_ax.spines['right'].set_color('tab:red')
                f_ax.spines['bottom'].set_color('tab:red')

            # actually plot data = mask, input data, or extracted data
            if plot_mask == 1:  # plot masks
                im = f_ax.imshow(np.flipud(mask[:, :, 0]), cmap=plt.cm.get_cmap('cubehelix_r', 10),
                                 aspect='auto', vmin=0, vmax=1, zorder=0)
            elif plot_mask == 2:  # plot extracted event
                im = f_ax.imshow(np.flipud(np.log10(np.abs(extracted_event))), cmap='afmhot_r', aspect='auto', vmin=-12,
                                 vmax=-9.5, zorder=0)
            else:  # plot input data, extracted_event=original data here
                im = f_ax.imshow(np.flipud(np.log10(np.abs(extracted_event))), cmap='viridis', aspect='auto', vmin=-11.5,
                                 vmax=-8.5, zorder=0)

            # string of  start time of each window
            start_time = str(self.utc_list[i][0].hour).zfill(2) + ':' + str(self.utc_list[i][0].minute).zfill(2)

            # plot start time with color visible on background
            if plot_mask == 1:  # dark on white background
                f_ax.set_title(start_time, x=0.25, y=0.65, fontsize=9, color='k')
            else:  # white on dark background
                f_ax.set_title(start_time, x=0.25, y=0.65, fontsize=9, color='white')

            # plotting details
            f_ax.set_xticks([])
            f_ax.set_yticks([])
            f_ax.set_xlim(0, 255)

            box_start = self.utc_list[i][0]  # start of window
            box_end = self.utc_list[i][1]  # end of window

            # mark events
            for ev_start in event_list_start:  # loop through event starts on day
                if box_start < ev_start[0] < box_end:  #  check if event start in window
                    min_boxstart = (ev_start[0] - box_start) / 60  # min of event after window start
                    f_ax.axvline(int(min_boxstart / 27.2 * 255), -1, 1, color='tab:green', lw=1, ls='dashed', alpha=0.9,
                                 zorder=2,label='MQS\nevent\nstart')  # plot event start marker
                    ev_type, ev_qual = self.check_label(ev_start)  # get event type and quality indicator
                    color_qual = ['tab:green', 'tab:orange', 'tab:red', 'tab:grey']  # colors for marker
                    f_ax.text(int(min_boxstart / 27.2 * 255) - 60, 250, ev_type, fontsize=10,
                              color=color_qual[ev_qual])  # ,fontweight='bold')  # add text marker to event

            for ev_end in event_list_end:  # loop through event ends on day, same as above
                if box_start < ev_end < box_end:
                    min_boxstart = (ev_end - box_start) / 60
                    f_ax.axvline(int(min_boxstart / 27.2 * 255), -1, 1, color='tab:orange', ls='dashdot', lw=1, alpha=0.9,
                                 zorder=2,label='MQS\nevent\nend')

            # annotate gaps, part 1
            if np.sum(np.isnan(self.sol_data[i, :, :, 0]))>1500:  #  check if many values = nan
                gapline = np.ones(256)*128
                # gapline = np.ma.masked_where(np.isnan(np.nanmean(sol_data[i,:,:,0],axis=0))==False, gapline)
                gapline = np.ma.masked_where(np.isnan(self.sol_data[i,100,:,0])==False, gapline)  # masks where data available
                f_ax.plot(gapline, lw=50, color='tab:gray', alpha=0.5,solid_capstyle="butt",label='Gap') # plot gap marker for nanas

            gap_start_list, gap_end_list = self.get_gaps_box(gaps_sorted, box_start, box_end)  # get gaps of stream
            # annotate gaps part 2
            if len(gap_start_list)>0:
                for p in range(len(gap_start_list)):  # loop through gaps
                    f_ax.plot([gap_start_list[p], gap_end_list[p]], [128, 128], lw=50, color='tab:gray', alpha=0.5,solid_capstyle="butt",label='Gap')
                    gap_windows.append([box_start, box_end])  # add gap marker

            # annotate lander activity, same as above
            lander_start, lander_end, act_annotations = self.get_gaps_box(lander_list_temp, box_start, box_end,return_label=True,min_secs=5)
            if len(lander_start)>0:
                for p in range(len(lander_start)):
                    annotation_colors = ['chocolate', 'mediumslateblue','gold']
                    f_ax.plot([lander_start[p], lander_end[p]], [128, 128], lw=50, color=annotation_colors[act_annotations[p]], alpha=0.25,
                              solid_capstyle="butt",label=['Arm','Calibr.','Recentr.'][act_annotations[p]])
                    gap_windows.append([box_start, box_end])

            # add lmst time to plot
            if i in [0, 22, 44, 66, 88]:  # indices corresponding to beginning of "blue rows"
                lmst_str = str(utc2lmst(box_start)[0].hour).zfill(2) + ':' + str(utc2lmst(box_start)[0].minute).zfill(
                    2) + '\nLMST'  # lmst time string
                f_ax.text(-15, 400, lmst_str, fontsize=10, color='tab:blue')  # add str

            if i == 0:  # add x and y label for first window
                f_ax.set_ylabel('0-10 Hz', fontsize=10)
                f_ax.set_xlabel('~27 mins', fontsize=10)
                f_ax.xaxis.set_label_position('top')
            # end of for loop going through 110 windows

        f_colorbar = fig.add_subplot(spec[15, :])  # add figure colorbar subplot
        cb = plt.colorbar(im, cax=f_colorbar, orientation='horizontal')  # add colorbar and details
        cb.ax.tick_params(labelsize=14)
        if plot_mask == 1:  # add label
            cb.set_label('Event mask values', fontsize=14)
        else:
            cb.set_label('Amplitude $[log_{10}(m/s/'+ r'\sqrt{Hz})]$', fontsize=14)

        # add title
        title_txt = 'Sol ' + str(self.sol) + " - " + str(self.utc_list[0][0])[:-8] + " to " + str(self.utc_list[-1][-1])[:-8]
        plt.suptitle(title_txt, y=0.91,color='k')

        # get detections of the two integrated masks time series
        if check_detections:
            lists_list = [event_list_start, event_list_end, list_susp, susp_list_end]
            detection_list = self.get_detections(all_band_sol1, all_band_sol2, lists_list, height=5, distance=100, threshold=self.threshold)
            # sort out detections if in window with gap
            detection_list = self.remove_gap_detections(detection_list, gap_windows)
        # self.detection_list = detection_list  # added later

        # extract events
        if classify:
            y_eventZ = (self.sol_data[:,:,:,0] + 1j*self.sol_data[:,:,:,1]) * self.y_window[:,:,:,0] # extract event
            y_eventN = (self.sol_data[:,:,:,2] + 1j*self.sol_data[:,:,:,3]) * self.y_window[:,:,:,2]
            y_eventE = (self.sol_data[:,:,:,4] + 1j*self.sol_data[:,:,:,5]) * self.y_window[:,:,:,4]
            y_allZ = self.sol_data[:,:,:,0] + 1j*self.sol_data[:,:,:,1]  # original data
            y_allN = self.sol_data[:,:,:,2] + 1j*self.sol_data[:,:,:,3]
            y_allE = self.sol_data[:,:,:,4] + 1j*self.sol_data[:,:,:,5]

            ev_class_list = []  # list to collect event class, amplitudes

        # extract info and plot stuff
        if plot_marker:  # plot marker of sign. events
            ax_list = fig.axes  # figure
            start_ls = []  # list for detections: start
            end_ls = []  # list for detections: end
            temp_detectval = []  # detection values

            for row in detection_list:  # loop through detections
                temp_detectval.append(row[0])  # collect detection value

                if not row[0]=='Missed event':  # skip rows with missed event info, otherwise append
                    start_ls.append(self.num2boxindex(row[5]))
                    end_ls.append(self.num2boxindex(row[6]))
                else:  # append for skipped rows
                    start_ls.append((999, 999, 999, 999))
                    end_ls.append((999, 999, 999, 999))

            for index in range(len(start_ls)):  # loop through detections
                row_start = start_ls[index]
                row_end = end_ls[index]

                if row_start[0]==999:  # skip if missed event
                    if classify:
                        ev_class_list.append(['','','','',''])  # append empty row for missed events and continue
                    continue

                if classify:  # classify events, get amplitude
                    # denoised amplitude
                    eventtype, eventampl = self.classify_detections(y_eventZ, y_eventN, y_eventE, start_ls[index], end_ls[index])
                    # original amplitude
                    _, eventampl_original = self.classify_detections(y_allZ, y_allN, y_allE, start_ls[index], end_ls[index])

                    ev_class_list.append([eventtype,eventampl[0],eventampl[1],eventampl_original[0],eventampl_original[1]])
                    ev_color =  self.get_ev_type_color(eventtype)

                    # mark detections - skip if below threshold (event family-dependent)
                    if eventtype=='LF/BB' and temp_detectval[index]<self.lf_threshold:
                        continue
                    if eventtype == 'HF/24' and temp_detectval[index] < self.hf_threshold:
                        continue
                    if eventtype == 'VF' and temp_detectval[index] < self.hf_threshold:
                        continue

                    # currently only used to classify lf and hf event family: types to event family here
                    if eventtype != 'LF/BB':
                        eventtype = 'HF family'
                    else:
                        eventtype = 'LF family'

                    ev_label = 'Predicted\n' + eventtype  # get event label (from prediction) for legend

                # plotting marker for detection, event family
                for i in range(2):  # add marker for two overlapping rows
                    box_start = row_start[0+i]
                    box_end = row_end[0+i]
                    int_start = row_start[2+i]
                    int_end = row_end[2+i]

                    if box_start is not None and box_end is not None:
                        if box_start==box_end: #start&end in same box
                            ax_list[box_start].plot([int_start,int_end],[15,15],color=ev_color,lw=10,alpha=0.75,
                                                    solid_capstyle="butt",label=ev_label)

                        if box_end > box_start:  #start and end in different box.
                            ax_list[box_start].plot([int_start,255],[15,15],color=ev_color,lw=10,alpha=0.75,
                                                    solid_capstyle="butt",label=ev_label)
                            ax_list[box_end].plot([0,int_end],[15,15],color=ev_color,lw=10,alpha=0.75,
                                                  solid_capstyle="butt",label=ev_label)

                            diff_boxes = box_end-box_start

                            if diff_boxes>2: # 1 or more boxes between start and end
                                for j in range(2,diff_boxes-1):
                                    xlim_temp = ax_list[box_end-j].get_xlim()
                                    ax_list[box_end-j].plot([xlim_temp[0], xlim_temp[1]], [15, 15], color=ev_color, lw=5, alpha=0.75,
                                                          solid_capstyle="butt",label=ev_label)
                    else:  # start or end on different sol
                        if box_start is None and box_end is not None:
                            ax_list[box_end].plot([0,int_end],[15,15],color=ev_color,lw=10,alpha=0.75,
                                                  solid_capstyle="butt",label=ev_label)

                        if box_end is None and box_start is not None:
                            ax_list[box_start].plot([int_start,255],[15,15],color=ev_color,lw=10,alpha=0.75,
                                                    solid_capstyle="butt",label=ev_label)

        if plot_legend:  # add figure legend
            lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
            lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

            unique = [(h, l) for i, (h, l) in enumerate(zip(lines, labels)) if l not in labels[:i]]

            leg = fig.legend(*zip(*unique),bbox_to_anchor=(1.01, 0.5),fontsize=10,framealpha=0.85)

            for legobj in leg.legendHandles:
                if legobj.get_linewidth()>5:
                    legobj.set_linewidth(5)
                else:
                    legobj.set_linewidth(2)


        # show figure or save
        if save_dir is not None:
            plt.savefig(save_dir + str(plot_mask) + '_sol_' + str(self.sol) + '.png', bbox_inches='tight',dpi=150)
            plt.close()
        else:
            plt.show()
        combined_list = []  # list combining detection_list and ev_class_list (ampl, eventtype,...)
        for i in range(np.shape(detection_list)[0]):
            combined_list.append([detection_list[i][0]]+[str(detection_list[i][1])]+detection_list[i][2:]+ev_class_list[i])
        # add to instance
        self.detection_list = detection_list
        self.combined_list = combined_list


    def get_events_sol(self,list_ev, return_name=False, include_overlap=False):
        """
        Get events on sol
        Input: list with events (sus signals), boolean values
        Output: two lists with event start and event end wiith event on this sol
        Requires: sol_span
        """
        # sol_span = sol_span_in_utc(utc2sol(self.utc_list[55][0]))  # get sol span

        sol_span = sol_span_in_utc(self.sol)  # get sol span

        if include_overlap:  # optional, include overlap to previous/following sol
            sol_start_utc = sol_span[0]-1800
            sol_end_utc = sol_span[1]+1800
        else:
            sol_start_utc = sol_span[0]
            sol_end_utc = sol_span[1]
        # create lists
        utc_start = []
        utc_end = []


        for utc in list_ev: # go through list with all events
            # check wether event start or end on sol
            cond1 = sol_start_utc < utc[0] < sol_end_utc
            cond2 = sol_start_utc < utc[1] < sol_end_utc
            if cond1 or cond2:
                # window start
                if return_name:
                    utc_start.append([utc[0], utc[2], utc[3], utc[4]])
                else:
                    utc_start.append([utc[0], utc[2], utc[3]])
                # window end
                utc_end.append(utc[1])

        return utc_start, utc_end

    def get_landeractivity_sol(self,list_time, include_overlap=False):
        """
        Get lander activity on sol
        Input: list with lander activity , boolean values
        Output: lists with lander activity start and end and type on this sol
        Requires: sol_span,
        """
        # follows get_events_sol
        sol_span = sol_span_in_utc(self.sol)
        if include_overlap:
            sol_start_utc = sol_span[0]-1800
            sol_end_utc = sol_span[1]+1800
        else:
            sol_start_utc = sol_span[0]
            sol_end_utc = sol_span[1]
        return_list = []
        for row in list_time:
            utc_start = UTCDateTime(row[0])
            utc_end = UTCDateTime(row[1])
            cond1 = sol_start_utc < utc_start < sol_end_utc
            cond2 = sol_start_utc < utc_end < sol_end_utc

            if cond1 or cond2:
                if row[2]=='arm':
                    type_temp=0
                elif row[2]=='calibration':
                    type_temp=1
                elif row[2]=='recentering':
                    type_temp=2
                # window start
                return_list.append([utc_start, utc_end, type_temp])

        return return_list

    def sort_gaps(self):
        """
        Sort gaps
        Input: list with gaps, extracted from sol stream
        Output: list with sorted gaps
        """
        sorted_gaps = []  # list with sorted gaps
        gaps = self.gaps
        if len(np.shape(gaps)) == 1:
            gaps = [gaps]
        elif len(np.shape(gaps)) > 1:
            for gap in gaps:  # loop through gaps, if available
                if np.abs(UTCDateTime(gap[4]) - UTCDateTime(gap[5])) > 120 and float(gap[6])>0:  # skip gaps <120s
                    gap_bool = True  # bool marker indicating if gap should be added to list
                    for sorted_gap in sorted_gaps:  # skip doubles ~ gaps very similar to others in sorted_gaps list
                        if np.abs(UTCDateTime(gap[4]) - UTCDateTime(sorted_gap[0])) < 3:
                            gap_bool = False
                    for sorted_gap in sorted_gaps:  # skip doubles
                        if np.abs(UTCDateTime(gap[5]) - UTCDateTime(sorted_gap[1])) < 3:
                            gap_bool = False
                    if gap_bool:  # append to list
                        sorted_gaps.append([gap[4], gap[5]])
        return sorted_gaps

    def find_grid(self,i):
        """
        Helper method to get row and colum for i-th windows
        Input: i
        Output: row and col for plot
        """
        block = int(i / 22)  # block = two overlapping rows
        if int(i % 2) == 0:  # upper row
            row = 0
        else:  # lower row
            row = 1
        row_grid = (2 * block + row)  # compute row

        # find col
        col_grid = int((i % 22) / 2) * 10
        if i < 22:
            row_grid = row_grid
        elif 21 < i < 44:
            row_grid = row_grid + 1
        elif 43 < i < 66:
            row_grid = row_grid + 2
        elif 65 < i < 88:
            row_grid = row_grid + 3
        else:
            row_grid = row_grid + 4
        return row_grid, col_grid

    def check_label(self,ev_start):
        """
        Helper method to shorten event type label
        Input: row of list with events
        Output: event type (BROADBAND=BB) and quality (ABCD = 0123)
        """
        # event name
        if ev_start[1] == 'LOW_FREQUENCY':
            event_type = 'LF'
        elif ev_start[1] == 'BROADBAND':
            event_type = 'BB'
        elif ev_start[1] == '2.4_HZ':
            event_type = '24'
        elif ev_start[1] == 'HIGH_FREQUENCY':
            event_type = 'HF'
        else:
            event_type = 'VF'
        # event quality
        if ev_start[2] == 'A':
            quality = 0
        elif ev_start[2] == 'B':
            quality = 1
        elif ev_start[2] == 'C':
            quality = 2
        else:
            quality = 3
        return event_type, quality

    def get_gaps_box(self,gaps_sorted, box_start, box_end, min_secs=30,return_label=False):
        """
        Helper method to check if gap in time window (box) and translate utc time to time window indices (0-255)
        Input: gaps_sorted list, start and end of box, optional inputs (return_label for lander act.)
        Output: list with gaps/activity translated to window indices
                """
        # check for gaps, init. bool markers
        gap_start = None
        gap_end = None
        activity_type = None
        gap_start_list = []  # init. lists
        gap_end_list = []
        activity_type_list = []

        for gap in gaps_sorted:  # loop through sorted gaps/lander act. on sol
            if np.abs(gap[0] - gap[1]) > min_secs:  # ignore gaps<min_secs
                # compute bool markers for different cases
                cond1 = box_start < gap[0] < box_end  # gap start within box
                cond2 = box_start < gap[1] < box_end  # box end within box
                cond3 = gap[0] < box_start < gap[1]  # box start in gap window
                cond4 = gap[0] < box_end < gap[1]  # box end in gap window
                # convert utc time to counting in time window (0-255)
                if cond1 and cond2:  # start and end in box
                    gap_start = (gap[0] - box_start) / 60
                    gap_start = int(gap_start / 27.3475 * 255)  # changed from 27.2, 256
                    gap_end = (gap[1] - box_start) / 60
                    gap_end = int(gap_end / 27.3475 * 255)

                elif cond1:  # only start in box
                    gap_start = (gap[0] - box_start) / 60
                    gap_start = int(gap_start / 27.3475 * 255)  # changed from 27.2, 256
                    gap_end = 256
                elif cond2:  # only end in box
                    gap_end = (gap[1] - box_start) / 60
                    gap_end = int(gap_end / 27.3475 * 255)
                    gap_start = 0
                elif cond3 and cond4:  # start before box start, end after box end
                    gap_start = 0
                    gap_end = 256

                if cond1 or cond2 or cond3 and cond4:  # add to list
                    gap_start_list.append(gap_start)
                    gap_end_list.append(gap_end)

                if return_label:  # optional, add label
                    activity_type = gap[2]
                    if cond1 or cond2 or cond3 and cond4:
                        activity_type_list.append(gap[2])

        if return_label:
            # return gap_start, gap_end, activity_type
            return gap_start_list, gap_end_list, activity_type_list
        else:
            # return gap_start, gap_end
            return gap_start_list, gap_end_list

    def get_detections(self,all_band_sol1, all_band_sol2, lists_list, height=5, distance=100, threshold=100):
        """
        Checks detections of sol
        and compares detctions to cat. events / susp. signals if provided by list
        Input: all_band_sol1,all_band_sol2 (time series integrating event masks)
        for both shifted sequences of windows;
        lists_list: list with 4 lists for cat. event start/end, susp. signal start/end
        Height, distance, threshold param. for selecting significant detections
        Output: list with found cat. events, found susp. signals, found uncat. signals, missed cat. events
        Requires: check_detections, find_peaks (scipy)
        """
        # flatten lists, concatenated bool arrays - full sol
        full_sol1 = np.array(all_band_sol1).flatten()
        full_sol2 = np.array(all_band_sol2).flatten()  # height = 5
        # combined list, account for 50% (128 time bins) shift of list1 and list2
        sol_combined = np.zeros(len(full_sol1) + 128)
        sol_combined[0:len(full_sol1)] += full_sol1
        sol_combined[128:128 + len(full_sol1)] += full_sol2

        # find peaks
        peaks, properties = find_peaks(sol_combined, height=height, threshold=None, distance=distance, rel_height=0.5)

        # check detections
        detections_checked = self.check_detect(sol_combined, peaks, lists_list, threshold=threshold)

        return detections_checked


    def check_detect(self,sol_combined, peaks, lists_list, threshold=100):
        """
        Checks detections found by predictor
        Sorts out detections below threshold with check_significance
        For each peak: checks whether peak lies within event window,
        if not, checks whether peaks lies within susp. signal window
        if not, new detection
        checks whether catalogued events were missed
        relict: this comparison is done later more precisely...
        Input: combined detection value time series, found peaks,
        and lists (nested list witth start/end of events/susp window)from get_detection, threshold
        peaks (as indices of sol_combined)
        Output: list with detected events/ susp. windows. / new detections / missed events
        requires: check_significance
        """
        detections_checked = []  # init.list

        sol_start = np.min(self.utc_list)  # sol start and end
        sol_end = np.max(self.utc_list)
        seconds_sol = sol_end - sol_start  # other values
        sol_array_len = len(sol_combined)

        # get peaks above threshold
        above, _ = self.check_significance(sol_combined, peaks, threshold=threshold)
        # unpack lists with events/susp. signals
        event_list_start, event_list_end, susp_list_start, susp_list_end = lists_list

        # event_list_start, event_list_end
        for row in above:  # loop through found peaks above threshold
            event_bool = False  # set bool marker to false for event
            susp_bool = False  # set bool marker to false for sus signal
            peak = row[2]  # get peak
            val_sum = row[3]  # peak value
            start_detect = row[0]  # start and end
            end_detect = row[1]

            peak_utc = sol_start + (peak / sol_array_len) * seconds_sol  # compute peak utc

            # check for catalogued MQS events TRUE POSITIVES
            for event_index in range(len(event_list_start)):  # loop through events
                ev_start = event_list_start[event_index][0]
                ev_end = event_list_end[event_index]

                if ev_start - 30 < peak_utc < ev_end + 30: # check whether peaks match mqs events
                    ev_type = event_list_start[event_index][1]
                    ev_quality = event_list_start[event_index][2]
                    ev_name = event_list_start[event_index][3]
                    event_bool = True  # set bool marker to True if detection matches event

                    if not any(ev_name in sl for sl in detections_checked):  # avoid double detections
                        detections_checked.append([val_sum, peak_utc, ev_name, ev_type, ev_quality,start_detect,end_detect,peak])

            # if peak not corresponding to event, check for MQS susp. signal
            if event_bool == 0:
                for susp_index in range(len(susp_list_start)):  # loop through susp. signals
                    susp_start = susp_list_start[susp_index][0]
                    susp_end = susp_list_end[susp_index]
                    if susp_start - 30 < peak_utc < susp_end + 30:
                        detections_checked.append([val_sum, peak_utc, "Susp. Signal", "", "",start_detect,end_detect,peak])
                        susp_bool = True  # set bool marker to True if detection matches sus signal

            # Remaining signal FALSE POSITIVES / NEW DETECTIONS (=false detections/not catalogued events)
            if (event_bool + susp_bool) == 0:
                detections_checked.append([val_sum, peak_utc, "Not in Catalogue", "", "",start_detect,end_detect,peak])
                # print('Uncatalogued signal found!')

        # check for missed events FALSE NEGATIVES:
        for event in event_list_start:
            if not any(event[3] in sl for sl in detections_checked):
                print('Missed event:', event[3])
                detections_checked.append(['Missed event', event[0], event[1], event[2], event[3],"","",""])
        return detections_checked

    def check_significance(self,data, peaks, threshold=100):
        """
        Checks significance of detections
        Input: sol_combined (time series detection values, peaks of detections
        Output: two lists with detections above and below threshold
        requires: find_start_end
        """
        detections = self.find_start_end(data, peaks)  # finds start and end of prediction
        above_threshold = []  # list for detections above threshold
        below_threshold = []  # below

        for detection in detections:  # loop through detections
            sum_window = np.sum(data[detection[0]:detection[1]])  # sum detection time series from start-end of detect.

            if sum_window > threshold:  # check whether above/below threshold
                above_threshold.append([detection[0], detection[1], detection[2], sum_window, data[detection[2]]])
            else:
                below_threshold.append([detection[0], detection[1], detection[2], sum_window, data[detection[2]]])

        # sort out double detections: two peaks for same detection, higher one picked
        # sort list by ~time
        above_threshold = sorted(above_threshold, key=lambda x: (x[3], x[4]), reverse=True)

        selected_detections = []
        for row in above_threshold:
            if len(selected_detections) == 0:  # if empty, add to list
                selected_detections.append(row)
            elif len(selected_detections) == 1:  # if not empty, check if detections match
                if row[0] != selected_detections[0][0]:
                    selected_detections.append(row)
            elif not row[0] in np.array(selected_detections)[1, :]:
                selected_detections.append(row)

        return selected_detections, below_threshold

    def find_start_end(self,data, peaks):
        """
        Finds approx. start/end of detection:
        Input: sol_combined (time series), peaks of detections
        Output: list with [windo_start, window_end, peak] for each peak
        """
        # part of detection value timeseries corresponding to one event can be interrupted by e.g. glitch
        # data is smoothed to remove short low values because of this
        data = savgol_filter(data, 11, 3)
        detection = []
        for peak in peaks:  # loop through peaks
            i_left = 0  #  beginning of time series
            val = 99
            while val > 0.1 and (peak + i_left) > 0:  # move from peak of detection to left and
                # check if values still different from 0, and inside time window
                val = data[peak + i_left]
                i_left -= 1

            i_right = 0  # end of timeseries
            val = 99
            while val > 0.1 and (peak + i_right) < len(data) - 2:  # move from peak to right and same as above
                val = data[peak + i_right]
                i_right += 1
            detection.append([peak + i_left, peak + i_right, peak])
        return detection

    def remove_gap_detections(self,detections, gap_start_end):
        """
        Remove detections matching gaps
        Input: list of detections, list of gaps
        Output: list of detections minus detections matching gaps
        """
        new_detection_list = []  # init.list
        for row in detections:  # loop through detection and check if matches gap
            no_gap = True  # gap bool marker
            for row_gap in gap_start_end:
                if row_gap[0] <= row[1] <= row_gap[1]:
                    no_gap=False  # set to False if there is a gap
            if no_gap:
                new_detection_list.append(row)
        return new_detection_list

    def num2boxindex(self,num):
        """
        Helper method
        Input: number
        Output: box start and end,of alternating rows (blue and red)
        """
        num1 = num  # number=time bin in window (0-255)
        num2 = num1 -128  #  equivalent inb shifted row
        box1 = int(num1/256) * 2  # box number
        box2 = int(num2/256) *2  +1  # box number shifted row

        box1_step = num1%256  # rest
        box2_step = num2%256  # rest

        # check if first or last box, set to None
        if num1<0 or box1>109:
            box1=None
            box1_step=None

        if num2<0 or box2>110:
            box2=None
            box2_step=None

        return box1, box2, box1_step, box2_step

    def classify_detections(self,y_eventZ, y_eventN, y_eventE, row_start,row_end):
        """
        Classifies events: using ZNE stft data
        Input: stft data for ZNE, row info
        Output: event type and amplitude
        Requires: get_ev_vals, classify_median_vals
        """
        peak_ampl = []  #  init. list for peak amplitudes
        for comp in range(3):  # loops through comp
            y_event = [y_eventZ, y_eventN, y_eventE][comp] # get component data

            for i in range(2):  # loops through shifted windows
                # get indices and check all kind of things in order to collect time-freq. bins corresponding to one event
                # if required, concat. multiple windows associated to one event
                # get box and index of event start and end
                box_start = row_start[0 + i]
                box_end = row_end[0 + i]
                int_start = row_start[2 + i]
                int_end = row_end[2 + i]
                if i==0:
                    box_marker = 0  # indicating if detections is in first/last box

                # Check if detection is in first or last box: bool value
                if box_start==None and box_end!=None:
                    ev_extract = y_event[box_end, :, 0:int_end]
                    box_marker=1
                if box_end==None and box_start!=None:
                    ev_extract = y_event[box_start, :, int_start:-1]
                    box_marker=2
                if box_start==None and box_end==None:
                    ev_extract = np.zeros((256,3))
                    box_marker=3

                if (box_start == None) + (box_end == None) < 1:  # check that detection is not in first/last box
                    cond_temp = box_end-box_start
                    if cond_temp == 0:  # if start and end in same box
                        ev_extract = y_event[box_start, :, int_start:int_end]  # get extracted event start to end
                    elif cond_temp == 2:  # if end one box after start box
                        ev1 = y_event[box_start, :, int_start:256]  # get extracted event start to end of box
                        ev2 = y_event[box_end, :, 0:int_end]  # get extracted event start of box to event end
                        ev_extract = np.concatenate((ev1, ev2), axis=1)  # combine results from two boxes
                    else:  # 1 or more boxes in between startbox and endbox
                        ev1 = y_event[box_start, :, int_start:256]  # get extracted event start to end of box
                        for box_counter in range(0, cond_temp - 2, 2):  # loop through 'full' boxes in between
                            ev_12 = y_event[box_start + box_counter, :, :]
                            ev1 = np.concatenate((ev1, ev_12), axis=1)
                        ev2 = y_event[box_end, :, 0:int_end]  # get extracted event start of box to event end
                        ev_extract = np.concatenate((ev1, ev2), axis=1)  # combine results from two boxes

                ev_extract = ev_extract

                if i == 0:  # extracted event for blue boxes, upper rows
                    ev_extract_combined = np.copy(ev_extract)
                else:  # extracted event for red boxes, lower rows; get 'average' event
                    # first/last box: arrays different size:

                    # CHECK SPECIAL CASES
                    if box_marker==1:  # first box
                        temp_array = np.zeros_like(ev_extract_combined)  # get array size as detection in upper box
                        len_extra = np.abs(ev_extract_combined.shape[1]-ev_extract.shape[1])  # difference length of upper and lower (shifted) box
                        temp_array[:,len_extra:]=ev_extract  # fill lower box with zeros before detection to make it equal size to detection in upper box
                        ev_extract = temp_array  # abs value

                    if box_marker==2:  # last box
                        temp_array = np.zeros_like(ev_extract) # get array size as detection in lower box
                        temp_array[:,:ev_extract_combined.shape[1]]=ev_extract_combined # fill upper box with zeros after detection to make it equal size to detection in lower box
                        ev_extract_combined = temp_array  # set array to 0 padded one

                    if box_marker==3:
                        if ev_extract.shape[1]>ev_extract_combined.shape[1]:
                            ev_extract_combined = ev_extract
                        else:
                            ev_extract = ev_extract_combined

                    # TimeFreq domain
                    if comp==0:  # compute amplitude values for z component, as maximum of both row solution, append to list
                        lf_ampl, hf_ampl = self.get_ampl(np.abs(ev_extract),np.abs(ev_extract_combined), percentile_val=90)
                        peak_ampl.append(lf_ampl)
                        peak_ampl.append(hf_ampl)

                    # compute average of both rows
                    ev_extract_combined += np.abs(ev_extract)
                    ev_extract_combined /= 2

                    # compute vertical and horizontal solution
                    if comp==0:
                        vertical = ev_extract_combined
                    if comp==1:
                        horizontal = ev_extract_combined
                    if comp==2:
                        horizontal += ev_extract_combined

                        # compute median values in different freq bands for vertical and horizontal solution
                        # not really used anymore - UPDATE
                        median_vertical = self.get_ev_vals(vertical)
                        median_horizontal = self.get_ev_vals(horizontal)

                        event_type = self.classify_median_vals(median_vertical, median_horizontal)

        return event_type, peak_ampl

    def get_ampl(self,array1,array2,percentile_val=90):
        """
        Get amplitude of detection
        Input: array1,array2 corresponding to the two solutions of overlapping time windows,
        percentile_val-percentile value to compute peka amplitude
        Output: low and high freq percentile amplitude, taken as maximum of both solutions
        """
        # Compute max as chosen percentile
        # for both solutions = two shifted rows
        # row 1/blue
        lf_max1 = np.percentile(array1[4:18,:],self.percentile_val)  # compute percentile value for lf frequent bins
        hf_max1 = np.percentile(array1[56:68,:],self.percentile_val) # compute percentile value for hf frequent bins
        # row 2/red
        lf_max2 = np.percentile(array2[4:18,:],self.percentile_val)  # compute percentile value for lf frequent bins
        hf_max2 = np.percentile(array2[56:68,:],self.percentile_val) # compute percentile value for hf frequent bins
        return np.max([lf_max1,lf_max2]), np.max([hf_max1,hf_max2])  # return maximum from row 1 or row 2 solutions

    def get_ev_vals(self,temp,percentile_val=75):
        """
        Get percentile values in different freq bands (relict)
        Input: stft data (temp), and ercentile_val
        Output: array with percentile values in lf 2.4 hf (w/o 2.4) and vf band
        """
        # compute percentile values in different frequency bands, lf, 2.4, hf (w/o 2.4) and vf
        freq_lp = np.percentile(temp[3:26,:],percentile_val)
        freq_mode = np.percentile(temp[39:90,:],percentile_val)
        freq_hf = np.percentile(temp[103:154,:],percentile_val)
        freq_vf = np.percentile(temp[154:230,:],percentile_val)

        return_array = ([freq_lp, freq_mode, freq_hf, freq_vf])  # combine
        return np.nan_to_num(return_array,0)  # return and correct potential nans

    def classify_median_vals(self,valsZ, valsNE):
        """
        Classifies extracted median values into event types
        Input: median values extracted with get_ev_vals, 4 median vals from lp, 2.4m ode, hf and vf bands,
        separately for vertical and horizontal
        Output: event type as LF/BB, HF/2.4, VF or ?? if inclear
        """
        combined = np.concatenate((valsZ,valsNE))  # comine values
        # check which value is largest (which freq has most median energy)
        if np.argmax(combined)==0 or np.argmax(combined)==4:
            ev_type = 'LF/BB'
        elif np.argmax(combined)==1 or np.argmax(combined)==5:
            ev_type = 'HF/24'
        elif np.argmax(combined)==3 or np.argmax(combined)==7:
            ev_type = 'VF'
        else:
            ev_type = '??'
        return ev_type

    def get_ev_type_color(self,ev_type):
        """
        Helper method to convert event type into color for plot
        Currently changed to distinguesh only event family
        Input: event type
        Output: color for plot
        """
        if ev_type=='HF/24' or ev_type=='VF' or ev_type=='??':
            return 'coral'
        if ev_type=='LF/BB':
            return 'cornflowerblue'
        # if ev_type=='VF':
        #     return 'lightgreen'
        # if ev_type=='??':
        #     return 'tab:gray'

# class to clean detection lists
class clean_list:
    """
    Class to work with detection_list output check_prediction.plot_detections
    Attributes: detection list
    Methods:
    clean_detectionlist: basic cleaning of lists, duplicate removal
    check_detection_list: sepating types of detections
    sortout_doubles: double removal (slow)
    """
    def __init__(self,detection_list):
        self.list = self.clean_detectionlist(detection_list)

    def clean_detectionlist(self,list_detection):
        """
        Clean list of detections, by removing double detections (on consecutive sols, around midnight)
        Input: detection list
        Output: cleaned detection list
        """
        sorted_list2 = []  # init. list
        # add found MQS events to list, w/o dublicates
        for row in list_detection:  # loop through detections
            if not row[0] == 'Missed event' and row[2] not in ['Not in Catalogue',
                                                               'Susp. Signal']:  # consider only mqs events
                bool_val = True
                for row2 in sorted_list2:
                    if np.abs(UTCDateTime(row[1]) - UTCDateTime(row2[1])) < 30:  # check for duplicates
                        bool_val = False
                if bool_val:
                    sorted_list2.append(row)

        # add found sus. signal to list, w/o dublicates
        for row in list_detection:  # loop through detections
            if row[2] == 'Susp. Signal':  # consider only sus signal detections
                bool_val = True
                for row2 in sorted_list2:
                    if np.abs(UTCDateTime(row[1]) - UTCDateTime(row2[1])) < 30:
                        bool_val = False
                if bool_val:
                    sorted_list2.append(row)

        # add found new detections to list, w/o dublicates
        for row in list_detection:  # loop through detections
            if row[2] == 'Not in Catalogue':  # consider only uncatalogued/new detections
                bool_val = True
                for row2 in sorted_list2:
                    if np.abs(UTCDateTime(row[1]) - UTCDateTime(row2[1])) < 30:
                        bool_val = False
                if bool_val:
                    sorted_list2.append(row)

        # add missed detections to list, w/o dublicates
        for row in list_detection:  # loop through detections
            if row[0] == 'Missed event':  # consider only missed events
                bool_val = True
                for row2 in sorted_list2:
                    if np.abs(UTCDateTime(row[1]) - UTCDateTime(row2[1])) < 30:
                        bool_val = False
                        # print(row)
                if bool_val:
                    sorted_list2.append(row)
        return sorted_list2

    def check_detection_list(self, sol_range,events_list=[],susp_list=[]):
        """
        Create separate lists for mqs events, sus signals, new detections and missed mqs events
        compared to list of events in sol range; detections and known signals are compared by checking time overlap
        Input: detection list
        Output: adds individual lists to instance
        """
        start_sol = sol_range[0]
        end_sol = sol_range[1]

        vals_correct, vals_susp, vals_new = [], [], [] # init. lists for detection values
        list_of_detections = []  # list with detection start, end, cat/sus/new
        # sols = []

        # Get detection start and end - no missed events
        new_list = self.list
        for detect in new_list:  # loop through detection list

            if detect[0] == 'Missed event':  #  skip if row includes missed event
                continue
            else:
                #  get event start and end in utc, based on utc of peak number of time bins from peak to start and end
                ev_start = UTCDateTime(detect[1]) - np.abs(detect[7] - detect[5]) * 6.36
                ev_end = UTCDateTime(detect[1]) + np.abs(detect[7] - detect[6]) * 6.36
                list_of_detections.append(detect[:8] + [ev_start, ev_end] + [detect[8]])  # append to list

        # Get mqs ev list of sol range
        list_event_range = [] # get all events in sol range
        list_eventnames_range = []  # event names
        for ev in events_list:
            if start_sol <= utc2lmst(ev[0])[1] <= end_sol or start_sol <= utc2lmst(ev[1])[1] <= end_sol:
                list_event_range.append(ev)
                list_eventnames_range.append(ev[4])

        # Get sus signal list of sol range
        list_sussignals_range = []
        for ss in susp_list:
            if start_sol <= utc2lmst(ss[0])[1] <= end_sol  or start_sol <= utc2lmst(ss[1])[1] <= end_sol:
                list_sussignals_range.append(ss)

        # Check if detection match catalogue events
        detect_evname_match = []  # eventnames of matching mqs events
        detect_ev_match = [] #  matching mqs events
        detect_sussignals_match = [] # eventnames of sus signals
        detect_new = [] #  matching sus signals
        double_detections = []  # list with dupilcated detections (around midnight)

        for detect in list_of_detections:  # loop through detections
            d_start = detect[8]  # detection start
            d_end = detect[9]  # detection end
            double_detect = 0  # init. counter
            double_detect_ss = 0  # init. counter

            ev_bool, ss_bool = False, False  # bool marker for event and sus signal

            # compare to events in sol range,
            for ev in list_event_range: # loop through events
                e_start = ev[0]
                e_end = ev[1]

                # compute overlap
                if not (d_end <= e_start or d_start >= e_end):
                    overlap_sec = np.min([d_end - d_start, d_end - e_start, e_end - e_start, e_end - d_start])
                    overlap_event = (100 * overlap_sec / (e_end - e_start)) # overlap in %

                    if 25 < overlap_event:  # overlap larger than 25%, save to lists
                        detect_evname_match.append(ev[4])
                        row_append = detect[:8] + [detect[-1]]  # info from detections
                        row_append[2] = ev[4]  # info matched event
                        row_append[3] = ev[2]
                        row_append[4] = ev[3]
                        detect_ev_match.append(row_append)
                        ev_bool = True  # set event bool marker to True
                        double_detect += 1  # count detections for this event

            if double_detect > 1:  # count if double detections
                double_detections.append(detect)

            # compare to sus signals in sol range, similar ot above
            if not ev_bool:  # if not matched with event
                for ss in list_sussignals_range:  # loop through sus signals
                    e_start = ss[0]
                    e_end = ss[1]

                    if not (d_end <= e_start or d_start >= e_end):  # compute overlap
                        overlap_sec = np.min([d_end - d_start, d_end - e_start, e_end - e_start, e_end - d_start])
                        overlap_detection = (100 * overlap_sec / (d_end - d_start))
                        overlap_event = (100 * overlap_sec / (e_end - e_start))
                        max_overlap = np.max([overlap_detection, overlap_event])
                        if 33 < max_overlap:
                            row_append = detect[:8] + [detect[-1]]  # info from detections
                            detect_sussignals_match.append(row_append)
                            ss_bool = True
                            double_detect_ss += 1

            if double_detect_ss > 1:
                double_detections.append(detect)

            if ev_bool + ss_bool == 0:
                detect_new.append(detect[:8] + [detect[-1]])

        # get missed events: difference of events in range and detected events
        missed_list = natsorted(list(set(list_eventnames_range).difference(detect_evname_match)))

        # print out some stats
        print('Total detections: ' + str(len(list_of_detections)))
        print('Detections maching MQS event:' + str(len(detect_ev_match)))
        print('Detections maching MQS sus sig:' + str(len(detect_sussignals_match)))
        print('- matching multiple MQS event/sus signals:' + str(np.shape(double_detections)[0]))
        print('Detections new:' + str(len(detect_new)))

        print('- sum (minus doubles): ' + str(
            (len(detect_ev_match) + len(detect_sussignals_match) + len(detect_new)) - (np.shape(double_detections)[0])))

        print('MQS events missed:' + str(len(missed_list)) + '/' + str(len(list_eventnames_range)))

        # init some lists
        missed_info, detect_info, new_detection, mqs_events, mqs_susp = [],[],[],[],[]

        vals_correct, vals_susp, vals_new = [], [], []
        # collect mqs info of missed events
        for row in missed_list:  # loop through missed events
            for ev in list_event_range:  # loop through mqs events in time range
                if ev[4] == row:
                    missed_info.append(['Missed event', '', ev[2], ev[3], ev[4], '', '', ''])

        # collect detection value and info of detected events
        for row in detect_ev_match:
            vals_correct.append(row[0])
            mqs_events.append(row)

        # collect detection value and info of sus signal
        for row in detect_sussignals_match:
            vals_susp.append(row[0])
            mqs_susp.append(row)

        # collect detection value and info of new detections
        for row in detect_new:
            vals_new.append(row[0])
            new_detection.append(row)

        # sort out doubles, print some info
        print('MQS events found: ', len(mqs_events))
        mqs_events = self.sortout_doubles(mqs_events)
        print('MQS events found - after duplicate removal: ', len(mqs_events))

        print('MQS events missed: ', len(missed_info))
        missed_info = self.sortout_doubles(missed_info, mqs_events)
        print('MQS events missed - after duplicate removal: ', len(missed_info))

        # add to instance: list of detected mqs events, sus signals, new detections, and missed events
        self.mqs_events = mqs_events
        self.mqs_susp = mqs_susp
        self.new_detection = new_detection
        self.missed_info = missed_info

        self.len_mqs_events = len(mqs_events)
        self.len_mqs_susp = len(mqs_susp)
        self.len_new_detection = len(new_detection)
        self.len_missed_info = len(missed_info)

    def sortout_doubles(self,list1,list2=None):
        """
        Sort out duplicates from list, keep detections with higher detection value (slow)
        Input: detection list (optional second list, if two lists are provided: overlap of list1 with list2 is removed)
        Output: lists withjout double
        """
        list_without_doubles = []  # init. list

        if list2 is None:
            list1 = sorted(list1, key=lambda x: (x[2], x[0]), reverse=True)   # sort list by detect value and time
            for row in list1:  # loop through list
                if not any(row[2] in sl for sl in list_without_doubles):  #  check if in new list, this is slow...
                    list_without_doubles.append(row)
        else:  # compare to other list
            for row in list1:
                if not any(row[4] in sl for sl in list2):
                    list_without_doubles.append(row)
        return list_without_doubles

# class to plot individual detections, overview plot
class detection_plot:
    """
    Class to create overview plots of detections
    Attributes: stft data of one sample / time window (+same norm), clip value (fixed)
    Methods:
    make_detection_plot: main method to create plots
    events_on_sol: get events on sol
    normalize_percentile: normalise one sample
    get_event_envelope: get event envelope of extrated event
    """
    def __init__(self):
        self.sol_data = np.zeros((1,256, 256, 6))
        self.sol_data_norm = np.zeros((1,256, 256, 6))
        self.clip_val = 50

    def make_detection_plot(self,inv,detection_row,model_even, model_odd, list_event, plot_envelopes=False,save_dir=None,shift_start=300):
        """
        Create overview plots of detections
        Input: inv (station xml file), row of detections list with time info, model even and odd, catalogue file, optional values
        plot_envelopes-option to plot envelopes in different freq bands; save_dir-if dir provided: saving plot,
        otherwise show plot; shift_start-start of used timewindow realtive to start/utc_time in seconds
        Output: produces plot
        Requires: events_on_sol: normalize_percentile, get_event_envelope
        """
        utc_time, _ = get_event_start(detection_row)  # start time from list

        utc_start = utc_time - shift_start  # set window start 300 s before detection start / input time
        # get seismic data for sol
        self.sol=utc2lmst(utc_time)[1]  # get sol of detection
        # seismic data
        st_original, gaps = seis_by_sol(sol=self.sol, inv=inv, channel='02.BH', remove='response',return_gaps=True,
                                        output='VEL', mask=False, keep_overlap=True, check4data=True)

        # events on sol
        ev_sol = self.events_on_sol(list_event)

        # copies of stream for 27mins and longer, overview plot
        st_temp = st_original.copy()  # stream cut to exact window
        st_temp.trim(utc_start-6.4,utc_start+1628+6.4)  # window stream

        st = st_original.copy()  # stream cut to overview time window

        st.trim(utc_start-900-120,utc_start+1628+900+120)  # overview stream

        if not 32807 < len(st_temp[0]) < 32827:  # check length of stream, missing data might lead to fail
            print('length different...', row)

        # prepare figure specs
        fig = plt.figure(figsize=[12, 8])
        spec = gridspec.GridSpec(ncols=6, nrows=8, figure=fig, height_ratios=[25,9,50,9,30,20,10,2])
        f_ax = fig.add_subplot(spec[0, 0:7])
        spec.update(wspace=0.05, hspace=0.0)
        plt.rcParams['font.size'] = 8

        # compute and plot overview stream (Top row)
        f0, t0, Zxx_noise0 = scipy.signal.stft(st.select(component='Z')[0].data, **kwargs) # stft
        cb3 = f_ax.pcolormesh(t0-900-120, f0, np.log10(np.abs(Zxx_noise0)),vmin=-11.5, vmax=-9, shading='auto') # plot
        f_ax.text(-850-120, 8.5, 'Z', fontsize=8, color='k', bbox=dict(facecolor='white', alpha=0.75))  # label

        cb = plt.colorbar(cb3,pad=0.01)
        cb.set_label('Amplitude\n'+ r'$[log_{10}(m/s/\sqrt{Hz})]$',fontsize=10)

        f_ax.axvline(0, 0, 1, lw=2,color='white', ls='dashed')
        f_ax.axvline(1628+0, 0, 1, lw=2, color='white', ls='dashed')

        f_ax.set_xlabel('[s]',fontsize=10,labelpad=-2)
        f_ax.set_ylabel('[Hz]',fontsize=10)

        # plot marker for events on sol
        ev_list = []
        for ev in ev_sol:  # loop through events on sol, add marker for events on sol
            if st[0].stats.starttime - 30 < ev[0] < st[0].stats.endtime + 30:
                sec_after = ev[0] - st[0].stats.starttime - 900 - 120
                f_ax.scatter(sec_after, 9, s=100, color='tab:green', marker='>', edgecolors='white')
                ev_list.append(ev[4])
            if st[0].stats.starttime - 30 < ev[1] < st[0].stats.endtime + 30:
                sec_after = ev[1] - st[0].stats.starttime - 900 - 120
                f_ax.scatter(sec_after, 9, s=100, color='tab:orange', marker='<', edgecolors='white')
                ev_list.append(ev[4])

        # Get ZNE stft data
        for j in range(3):  # loop through components
            # get data, compute stft and add to sol_data
            comp = ['Z', 'N', 'E'][j]
            data = st_temp.select(component=comp)[0].data
            f, t, Zxx_noise = scipy.signal.stft(data, **kwargs)
            # print(np.shape(Zxx_noise))
            Zxx_noise = Zxx_noise[:, 1:-1]
            # Zxx_noise[238:, :] = 0 + 0j
            # add to instance
            self.sol_data[0, :, :, j * 2] = Zxx_noise.real
            self.sol_data[0, :, :, j * 2 + 1] = Zxx_noise.imag
            # normalise and add to instance
            self.sol_data_norm[0, :, :, j * 2: j * 2 + 2] = self.normalize_percentile(self.sol_data[0, :, :, j * 2: j * 2 + 2],
                                                                            limit=self.clip_val)

        # Select model and make prediction  - using 'even-sol model' for odd-sols and v.v.
        if self.sol % 2 == 1:  # odd sol, use model trained on even-sols
            self.y_window = model_even.predict(self.sol_data_norm)
        else:  # even sol, use model trained on odd-sols
            self.y_window = model_odd.predict(self.sol_data_norm)

        # plotting event masks Z (middle row, center right), details
        f_ax = fig.add_subplot(spec[2, 3:6])
        cb3 = f_ax.pcolormesh(t[1:-1], f, self.y_window[0, :, :, 0], vmin=0, vmax=1, cmap=plt.cm.get_cmap('cubehelix_r', 10),
                              shading='auto')  # plot predicted mask
        cb = plt.colorbar(cb3, ax=f_ax, pad=0.01)
        cb.set_label('Event mask', fontsize=10)
        f_ax.set_xlabel('[s]', fontsize=10, labelpad=-2)

        # check for events in time window
        for ev in ev_sol:
            if ev[0]-180 < utc_time+30 < ev[1] + 180:
                f_ax.set_title(ev[2] + ' ' + ev[3] + ' ' + ev[4], y=0.9, fontsize=8)

        f_ax.text(50, 9.25, 'Mask Z', fontsize=8, color='k', bbox=dict(facecolor='white', alpha=0.75))

        if plot_envelopes:  # optional, add envelopes in different bandwidths to z mask plot
            # mean_amplitude = self.get_event_envelope(sol_data_norm[0, :, :, 0] + 1j * sol_data_norm[0, :, :, 1],
            #                                     y_window[0, :, :, 0])  # used to plot extracted event envelope
            stalta_lp, stalta_2p4, stalta_4hz = self.get_stalta(st)

            x_sec = np.linspace(0,t[-1],len(stalta_lp))
            f_ax.plot(x_sec,4*stalta_lp+6,color='tab:blue',lw=1.5,label='0.2-0.8Hz',alpha=0.3)
            f_ax.plot(x_sec,4*stalta_2p4+6,color='tab:orange',lw=1.5,label='2.2-2.8Hz',alpha=0.3)
            f_ax.plot(x_sec,4*stalta_4hz+6,color='tab:green',lw=1.5,label='3.7-4.3Hz',alpha=0.3)

            f_ax.text(1400,5.4,'3.7-4.3 Hz', color='tab:green')
            f_ax.text(1400,4.9,'2.2-2.8 Hz', color='tab:orange')
            f_ax.text(1400,4.4,'0.2-0.8 Hz', color='tab:blue')

        f_ax.set_xlim(0, t[-1])

        # add subplots for NE event masks (middle row, right)
        # N masks
        pos1 = f_ax.get_position()
        f_nmask = fig.add_axes(
            [pos1.x0 + 1.12 * pos1.width, pos1.y0 + 0.75 * pos1.height, 0.25 * pos1.width, 0.25 * pos1.height])
        f_nmask.pcolormesh(t[1:-1], f, self.y_window[0, :, :, 2], vmin=0, vmax=1, cmap=plt.cm.get_cmap('cubehelix_r', 10),
                           shading='auto')

        # E mask
        f_emask = fig.add_axes(
            [pos1.x0 + 1.12 * pos1.width, pos1.y0 + 0 * pos1.height, 0.25 * pos1.width, 0.25 * pos1.height])
        f_emask.pcolormesh(t[1:-1], f, self.y_window[0, :, :, 4], vmin=0, vmax=1, cmap=plt.cm.get_cmap('cubehelix_r', 10),
                           shading='auto')
        # plotting: labels ...
        f_nmask.text(1400, 7, 'N', fontsize=10)
        f_emask.text(1400, 7, 'E', fontsize=10)

        f_nmask.set_xlabel('27min', fontsize=6)
        f_nmask.set_ylabel('0-10Hz', labelpad=-10, fontsize=8)
        f_emask.set_xlabel('27min', fontsize=6)
        f_emask.set_ylabel('0-10Hz', labelpad=-10, fontsize=8)

        f_nmask.axes.get_xaxis().set_ticks([])
        f_nmask.axes.get_yaxis().set_ticks([])
        f_emask.axes.get_xaxis().set_ticks([])
        f_emask.axes.get_yaxis().set_ticks([])

        # plot input data (middle row, left)
        f_ax = fig.add_subplot(spec[2, 0:3])

        plot_input = self.sol_data_norm[0, :, :, 0]
        cb1 = f_ax.pcolormesh(t[1:-1], f, plot_input, vmin=-3, vmax=3, cmap='bwr', shading='auto')  # plot norm. input
        # colorbar and details
        cb = plt.colorbar(cb1, ax=f_ax, pad=0.01)
        cb.set_label('Input normalized (linear)', fontsize=10)
        f_ax.text(50, 9.25, 'Input Z (real)', fontsize=8, color='k', bbox=dict(facecolor='white', alpha=0.75))

        f_ax.set_xlim(0, t[-1])
        f_ax.set_xlabel('[s]', fontsize=10, labelpad=-5)
        f_ax.set_ylabel('[Hz]', fontsize=10)

        # ZNE subplots  (bottom row, left, middle, right)
        for j in range(3):   # loop through zne components
            comp = ['Z', 'N', 'E'][j]
            # compute stft for all comp. for log plot, with longer winlen
            data = st_temp.select(component=comp)[0].data

            f2, t2, Zxx_noise2 = scipy.signal.stft(data, **kwargs)  # compute stft

            plot_label = [0, 2, 4][j]

            # adjust amplitude range in plot, relative to Z comp data
            if j == 0:
                f_index = find_freq(f2, 9)  # from Functions_MQNet
                plot_min = np.log10(np.percentile(np.abs(Zxx_noise2[:f_index, :]), 1))
                plot_max = np.log10(np.percentile(np.abs(Zxx_noise2[:f_index, :]), 99))


            # plot 1-10 on linear freq axis
            f_ax_up = fig.add_subplot(spec[4, plot_label:plot_label + 2])
            _ = f_ax_up.pcolormesh(t2, f2, np.log10(np.abs(Zxx_noise2)), vmin=plot_min, vmax=plot_max,
                                        shading='auto')  # ,cmap='jet')
            # plotting details
            f_ax_up.set_ylim(1, 10)
            f_ax_up.set_yscale('linear')
            f_ax_up.spines['bottom'].set_visible(False)
            if j != 0:
                f_ax_up.yaxis.set_major_formatter(NullFormatter())
            f_ax_up.xaxis.set_major_formatter(NullFormatter())
            if j == 0:
                f_ax_up.set_ylabel('[Hz]', fontsize=10)

            # plot 0.1-1 log freq. axis
            f_ax_low = fig.add_subplot(spec[5, plot_label:plot_label + 2])
            cb_log = f_ax_low.pcolormesh(t2, f2, np.log10(np.abs(Zxx_noise2)), vmin=plot_min, vmax=plot_max,
                                         shading='auto')  # ,cmap='jet')
            # plotting details
            f_ax_low.set_ylim(0.1, 1)
            f_ax_low.set_yscale('log')
            f_ax_up.text(50, 8.9, comp, fontsize=8, color='k', bbox=dict(facecolor='white', alpha=0.75))
            f_ax_low.spines['top'].set_visible(False)
            f_ax_low.set_xlabel('[s]', fontsize=10, labelpad=1)

            if j != 0:
                f_ax_low.yaxis.set_minor_formatter(FormatStrFormatter(""))
                f_ax_low.yaxis.set_major_formatter(FormatStrFormatter(""))
            else:
                f_ax_low.yaxis.set_minor_formatter(FormatStrFormatter(""))

        # get time stamps for name and title
        date_time = str(utc_start.date) + ' ' + str(utc_start.time)[:8]
        lmst_temp = utc2lmst(utc_start)[0]  # lmst
        lmst_time = str(lmst_temp.hour).zfill(2) + ':' + str(lmst_temp.minute).zfill(2)

        # make title
        title_txt = 'Window start: ' + date_time + ' (UTC) -  Sol: ' + str(utc2lmst(utc_time)[1]) + ' ' + lmst_time + ' (LMST)'

        plt.suptitle(title_txt, y=0.91, x=0.45, fontsize=12)

        # add common colorbar in bottom
        f_ax = fig.add_subplot(spec[7, 0:6])
        cb = plt.colorbar(cb_log, cax=f_ax, orientation='horizontal')
        cb.set_label('Amplitude ' + r'$[log_{10}(m/s/\sqrt{Hz})]$', fontsize=10)

        lmst_start, sol_start = utc2lmst(utc_start)  #
        if save_dir is not None:  # save if dir provided
            name = 'Sol' + str(sol_start) + '_' + str(lmst_start.hour).zfill(2) + str(lmst_start.minute).zfill(2)
            plt.savefig(save_dir + name + '.jpg', bbox_inches='tight')
            plt.close()
        else:  #  otherwise show
            plt.show()
        plt.rcParams['font.size'] = 12

    def events_on_sol(self, list_eventtime):
        """
        Find catalogued events (and susp. signals) on sol
        input: list with events
        return: list with events on sol (start/end)
        requires: sol_span_in_utc
        """
        events_list = []
        utc_range = sol_span_in_utc(self.sol)
        for event in list_eventtime:
            if utc_range[0] < event[0] < utc_range[1] or utc_range[0] < event[1] < utc_range[1]:  # check if event start/end within sol
                events_list.append(event)
        return events_list

    def normalize_percentile(self,data,quantile_range=(25,75),unit_variance=False,limit=50,clipping='constant'):
        """
        Normalizes individual samples with Sklearn Robustscaler (not batches)
        For data with outliers; separately for real and imag.
        Input: data, optional settings
        Output: norml. data
        Requires: Sklearn Robustscaler
        """
        data_real = data[:,:,0]
        data_imag = data[:,:,1]

        scaler =  RobustScaler(unit_variance=unit_variance)
        data_real_norm = scaler.fit_transform(np.expand_dims(data_real.flatten(),1)).reshape((256,256))
        data_imag_norm = scaler.fit_transform(np.expand_dims(data_imag.flatten(),1)).reshape((256,256))

        # clip high values
        stretch = limit/200
        if clipping=='constant':
            data_real_norm[data_real_norm>limit]=limit
            data_real_norm[data_real_norm<-limit]=-limit
            data_imag_norm[data_imag_norm>limit]=limit
            data_imag_norm[data_imag_norm<-limit]=-limit
        elif clipping=='tanh':
            data_real_norm = np.tanh(stretch*data_real_norm)
            data_imag_norm = np.tanh(stretch*data_imag_norm)
        elif clipping=='sigmoid':
            data_real_norm = (1/(1 + np.exp(-stretch*data_real_norm)))*2-1
            data_imag_norm = (1/(1 + np.exp(-stretch*data_imag_norm)))*2-1
        elif clipping=='arctan':
            data_real_norm = 2/np.pi * np.arctan(stretch*data_real_norm)
            data_imag_norm = 2/np.pi * np.arctan(stretch*data_imag_norm)
        else:
            data_real_norm = data_real_norm
            data_imag_norm = data_imag_norm

        data_return = np.zeros(data.shape)
        data_return[:,:,0] = data_real_norm
        data_return[:,:,1] = data_imag_norm

        return data_return

    def get_stalta(self,st):
        """
        Compute STA/LTA in different freq bands
        Input: stream
        Output: normalised STA/LTA ratio at long periods, 2.4Hz and 4Hz mode
        Requires: classic_sta_lta from obspy.signal.trigger
        """
        # copy stream, apply bandpass filter and compute sta/lta for different frequency bands
        st_lp = st.copy()
        data_lp = st_lp.select(component='Z').filter('bandpass',freqmin=.2,freqmax=.8,corners=6)[0].data

        st_2p4 = st.copy()
        data_2p4 = st_2p4.select(component='Z').filter('bandpass',freqmin=2.2,freqmax=2.8,corners=6)[0].data

        st_4hz = st.copy()
        data_4Hz = st_4hz.select(component='Z').filter('bandpass',freqmin=3.7,freqmax=4.3,corners=6)[0].data

        # extract time range of plot
        stalta_lp = classic_sta_lta(data_lp,100*20,1000*20)[20400:-20400]
        stalta_2p4 = classic_sta_lta(data_2p4,100*20,1000*20)[20400:-20400]
        stalta_4hz = classic_sta_lta(data_4Hz,100*20,1000*20)[20400:-20400]
        # clip high values
        stalta_lp[stalta_lp>3]=3
        stalta_2p4[stalta_2p4>3]=3
        stalta_4hz[stalta_4hz>3]=3

        # normalise  by max of three freq bands
        max_val = np.max([stalta_lp,stalta_2p4,stalta_4hz])
        stalta_lp /=max_val
        stalta_2p4 /=max_val
        stalta_4hz /=max_val
        return stalta_lp, stalta_2p4, stalta_4hz

    def get_event_envelope(self,Zxx,event_mask):
        """
        Compute envelope of extracted event
        Input: stft and predicted event mask
        Output: max-normalised envelope
        """
        # compute mean envelope of extracted event stft
        event_extracted = np.abs(Zxx * event_mask)
        mean_ampl = np.nanmean(event_extracted,axis=0)
        return mean_ampl/np.max(mean_ampl)


