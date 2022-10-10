"""
Script to use MarsQuakeNet on seismic data for event detection
load packages from Package_MQNet_JGRPlanets.py
requires  data and folder structure as define in zipn
Developed for MarsQuakeNet / InSight data.
Author: N.Dahmen - ETH Zurich
Environment: conda env create --name environment_name -f environment_droplet.yml
"""

import os.path
import time
import numpy as np
from obspy import UTCDateTime
import tensorflow as tf
# %% change to dir of Code
try:
    os.chdir('/home/niko/Schreibtisch/ML/Data/Revision/Code/')
except:
    print(os.getcwd())

# %% Loading classes from Package_MQNet_JGRPlanets
# predict_masks: Class to get data and make predictions, sol-wise
# check_prediction: class to check predictions, extract detections etc. (uses predict_mask instance)
# clean_list: class to clean detection lists (uses list produced by check_predictions)
# detection_plot: class to plot individual detections, overview plot (uses cleaned list from clear_list)
# See help(predict_masks), help(check_prediction), help(clean_list), help(detection_plot)
# load packages from local path, or adjust
sys.path.insert(1, './')
from Package_MQNet_JGRPlanets import predict_masks, check_prediction, clean_list, detection_plot
# %%
# load inventory file, station response (not required for preprocessed example, Sol 923))
# Dataless can be downloaded from: https://www.seis-insight.eu/en/science/seis-data/seis-metadata-access
# InSight Mars SEIS Data Service. (2019). SEIS raw data, Insight Mission. IPGP, JPL, CNES, ETHZ, ICL, MPS, ISAE-Supaero, LPG, MFSC.
# https://doi.org/10.18715/SEIS.INSIGHT.XB_2016
# inv = obspy.read_inventory('./Data/<PUT INV FILE HERE>')

# load MQS catalogue used in paper
list_temp = np.loadtxt('./Data/Lists/mqs_event_list.csv',delimiter=',',dtype='str')
list_eventtime = []  # fix this
for row in list_temp:
    list_eventtime.append([UTCDateTime(row[0]),UTCDateTime(row[1]),row[2],row[3],row[4]])

# %% LOAD MarsQuakeNet models

model_even = tf.keras.models.load_model('./Data/Models/model_even_v1_2.h5')
model_odd = model_even  # for testing
# model_odd = tf.keras.models.load_model('./Data/Models/model_odd_v1_1.h5')

# model_even.summary()
# %%
########################################################################################################################
#  Run detection, sol-wise
########################################################################################################################
# tool to run sol-wise prediction and collect detections
# define sol range to check; data (preprocessed) provided for Sol 923
start_sol = 923#182
end_sol = 923#1224
sol_range = [start_sol,end_sol]
save_dir = './Data/Output/Prediction/'  # dir to save plots to

detection_list_sols, skipped_sols = [], [] # list to add detections, skipped sols
start_time = time.time()  # timer

for sol in range(start_sol,end_sol+1):  # loop through sols as defined
    print('Processing sol: ',sol)  # SEIS data preprocessing takes time...
    try:
        # preprocess data using predict_mask class
        sol_instance = predict_masks()  # create prediction instance
        sol_instance.get_stft_sol(sol, None)  # preprocess seismic data, add stft inputs to instance; replace None with inv
        sol_instance.make_predictions(model_even, model_odd)  # add predicted masks to instance
        # extract detections, plot using check_prediction class
        sol_prediction = check_prediction(sol_instance)  # create sol check_prediction instance
        sol_prediction.plot_detections(list_ev=list_eventtime,list_susp=[], list_lander=[],save_dir=None) # provide dir to save
        detection_list_sols += sol_prediction.combined_list  # add detections to list
    except:
        print('skipped Sol ' + str(sol))
        skipped_sols.append(sol)

print("--- Processing time %s [min]  ---" % ((time.time() - start_time)/60))
print('skipped Sols:')
print(skipped_sols)

# %% Class to clean list, e.g. remove double detections around midnight of two consecutive sols
clean_list_object = clean_list(detection_list_sols)  # get class instance to clean detection lists
clean_list_object.check_detection_list(sol_range,events_list=list_eventtime,susp_list=[])  # clean list, separate them and save as instance

# products
new_list = clean_list_object.list  # list with all detections
mqs_events = clean_list_object.mqs_events  # list with detections matching MQS events
mqs_susp = clean_list_object.mqs_susp  # list with detections matching suspicious signals, if list provided
new_detection = clean_list_object.new_detection  # list with only new detections
missed_info = clean_list_object.missed_info  # list with false negative events

# save detection list
dir_save = './Data/Lists/ListSol'+str(start_sol) +'_'+str(end_sol)+'.csv'
# np.savetxt(dir_save,new_list,delimiter =",",fmt ='% s')
print(np.shape(detection_list_sols))

# %% Produce overview plots of individual detections from lists (produced above)
save_dir = './Data/Output/Detections/'
list_of_detections = new_list  # mqs_events, new_detection if interested in specific detections
for i in range(len(list_of_detections)):
    try:
        detect_plot = detection_plot()  # create plot instance
        # produce plot
        detect_plot.make_detection_plot(None,list_of_detections[i],model_even, model_odd, list_eventtime, plot_envelopes=False, save_dir=None)
    except:
        print('Skipped:')
        print(list_of_detections[i])

# %%




























































