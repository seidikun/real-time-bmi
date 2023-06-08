# real-time-bmi

## General Setup
- Clone/Download this project
- Create local data folder (e.g. ./Data/)
- Download OpenVibe v3.x
- Download python 3.7

## How to use processing

### mi processing v1
#### Configuration
- Open openvibe.conf in text editor and set Path_Project to the folder you choose to save the project, and Path_save to above created folder
- Substitute OpenVibe config file (original openvibe.conf) in e.g. C:\Program Files\openvibe-3.4.0-64bit\share\openvibe\kernel with above edited openvibe.conf

#### Using environments
- If new subject, create a new subject in Path_Save
- Open C:\Program Files\openvibe-3.4.0-64bit\share\openvibe\kernel\openvibe.conf and edit Participant and Session_nb
- Run mi processing v1 environments in sequence

### mi processing v2
#### Configuration
- Open config.txt in text editor and set Path_Project to the folder you choose to save the project, and Path_save to above created folder
- Pip install pyriemann

#### Using environments
- Use environment 16chan_1_acquisition to get training data
- Open python IDE and run 16chan_2_train_classifier_ts.py
- Online Aplication, use environment 16chan_3_online
