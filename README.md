# MarsQuakeNet
### Package to train and use MarsQuakeNet (MQNet), JGR Planets (2022JE007503)
##### "MarsQuakeNet: A More Complete Marsquake Catalogue Obtained by Deep Learning Techniques"
##### Paper: [doi: 10.1029/2022JE007503](https://doi.org/10.5281/zenodo.7157843)
##### Supplemental Material: [doi: 10.5281/zenodo.7157843](https://doi.org/10.5281/zenodo.7157843)
 

![Image](/Data/Output/MethodOverview.jpg)


##### Includes:
- Script_MQNet_JGRPlanets.py: main script to use MarsQuakeNet on InSight 20sps VBB data for event detection: produces list with detections, full-sol prediction plots ( Fig7 a,b), detection overview plot (e.g. S8);
- Package_MQNet_JGRPlanets.py: classes used in (Script_MQNet_JGRPlanets.py)
- environment_droplet.yml: dependencies
- ModelTraining/TrainingMQNet_JGRPlanets.ipynb: Colab notebook with UNet architecture implementation for model training
- ModelTraining/Requirements_TrainingMQNet_JGRPlanets.txt: dependencies


##### Requires: 
 - Waveform data (provided for Sol 923)
 - Station inventory (see below) 
 - MQS event list
 - Trained model
 - Assumes folder structure as described below 
 
 

 ##### Find data and folder structure at [MarsQuakeNetCode.zip](https://doi.org/10.5281/zenodo.7157843)): 
 

###### Folder structure:

```
+-- Script_MQNet_JGRPlanets.py: script to use MarsQuakeNet
+-- Package_MQNet_JGRPlanets.py: classes used in Script_MQNet_JGRPlanets.py
+-- environment_droplet.yml: dependencies
+-- ModelTraining: 
|   +-- TrainingMQNet_JGRPlanets.ipynb: Jupyter notebook for model training
|   +-- Requirements_TrainingMQNet_JGRPlanets.txt: dependencies
+-- Data: folder where input/output data are stored
|   +-- MethodOverview_mod.jpg (image shown above)
|   +-- Models
|   |   +-- model_even_v1_2.h5 (even-sol model)
|   +-- Lists
|   |   +-- mqs_event_list.csv
|   |   +-- ListSol923_923.csv
|   +-- Output
|       +-- Detections 
|           +-- ...
|       +-- Prediction 
|           +-- ...
|   +-- Waveforms
|       +-- 02.BH:
|           +-- 2018:  folder for raw waveform data
|           +-- 2019:  folder for raw waveform data  
|           +-- 2020:  folder for raw waveform data
|           +-- 2021:  folder for raw waveform data
|           +-- 2022:  folder for raw waveform data
|           +-- PreProcessed:  folder for preprocessed data
|               +-- Gaps:   folder with csv files for gaps
|                   +-- gaps_sol923.csv
|               +-- SEIS:   folder with preprocessed seis data
|                   +-- XB.ELYSE.02.BHZNE.Sol0923.mseed


