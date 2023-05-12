# -*- coding: utf-8 -*-
"""

"""

# Imports
import pickle
import random
import numpy  as np
import pandas as pd

from scipy                         import signal
from matplotlib                    import pyplot as plt
from sklearn.metrics               import cohen_kappa_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model          import LogisticRegression
from pyriemann.estimation          import Covariances
from pyriemann.tangentspace        import tangent_space
from pyriemann.utils.viz           import plot_embedding
from pyriemann.utils.mean          import mean_covariance

# Riemannian Tangent Space Classifier Training

#################### TROQUE O NOME DO ARQUIVO AQUI ###########################
filename = 'C:\\Users\\Laboratorio\\Desktop\\motor_mi_seidi_6.csv'

# Signal Epoching

# Define Parameters
overlap_step = 0.1  # tamanho do passo para sobreposição de janelas
max_trial_t  = 3.75 # tamanho máximo de uma janela de tentativa

# Read csv training file
df_me        = pd.read_csv(filename)

# Indexes for Left MI and Right MI, respectively
event_id     = pd.to_numeric(df_me['Event Id'], errors='coerce').fillna(0).astype(np.int64)
left_inds    = event_id.index[event_id == 769].tolist()
right_inds   = event_id.index[event_id == 770].tolist()
  
# Sampling frequency
Fs           = 1/np.mean(np.diff(df_me['Time:512Hz']))
b, a         = signal.butter(4, (8,30), 'bandpass', fs=Fs)

# Get EEG data from csv file and band-pass filter
data_eeg     = df_me.iloc[:, 2:18].values
data_eeg     = signal.filtfilt(b, a, data_eeg.T).T
    
# Epoching size
trange       = np.arange(int(Fs*0.5),int(Fs*2.5), 4) # Parameter 4 downsamples the signal in 4 times (512Hz to 128Hz)
  
# Window shift trhough trials, this does data augmentation
while trange[-1] < int(max_trial_t*Fs):        

  # Epochs for Left MI
  ts            = [i + trange for i in left_inds]
  epochs_left   = data_eeg[ts,:]
  labels        = [0 for i in left_inds]
  trial_num     = [i for i in range(len(left_inds))]

  # Epochs for Right MI
  ts            = [i + trange for i in right_inds]
  epochs_right  = data_eeg[ts,:]
  epochs_mne    = np.vstack([epochs_left, epochs_right])
  labels        = np.append(labels, [1 for i in right_inds])
  trial_num     = np.append(trial_num, [i for i in range(len(right_inds))])

  # Concatenate data
  if 'data' not in locals():
      data          = epochs_mne
      all_labels    = labels
      all_trial_num = trial_num
  else:
      data          = np.vstack([data, epochs_mne])
      all_labels    = np.append(all_labels, labels)
      all_trial_num = np.append(all_trial_num, trial_num)

  # New window
  trange += int(overlap_step*Fs)

# Randomization
p             = np.random.permutation(len(all_labels))
data          = np.swapaxes(data[p,:,:,],1,2)
all_labels    = all_labels[p]
all_trial_num = all_trial_num[p]

## Cross-validation

# Init variables
best_accuracy = 0
best_kappa    = 0
best_c_mean   = None
n_folds       = 20 # Cross-validaiton folds

# Array of unique labels for each trial
unique_labels = np.unique(all_trial_num)

# Arrays of accuracies and kappas
accuracies    = np.zeros(n_folds)
kappas        = np.zeros(n_folds)

print('Running model cross-validation...')
for fold in range(n_folds):

    # Randomly choose 2 trial labels
    val_labels = random.sample(list(unique_labels), 2)

    # Training and validation indexes
    train_inds           = np.where(np.logical_not(np.isin(all_trial_num, val_labels)))[0]
    val_inds             = np.where(np.isin(all_trial_num, val_labels))[0]

    # EEG data epochs to cov matrices
    cov_data_train       = Covariances().transform(data[train_inds])
    cov_data_val         = Covariances().transform(data[val_inds])
    
    # Taining and validation split
    X_train, y_train     = cov_data_train, all_labels[train_inds]
    X_val, y_val         = cov_data_val, all_labels[val_inds]
    
    # Mean convariance matrix
    C_mean               = mean_covariance(X_train)
    
    # Project covariance matrices to euclidean tangent plane
    tan_space_covs_train = tangent_space(X_train, C_mean)
    tan_space_covs_val   = tangent_space(X_val, C_mean)
    
    # Train logistic regression classifier
    lr = LogisticRegression(random_state=0)
    lr.fit(tan_space_covs_train, y_train)
    
    # Current fold accuracy
    accuracy = lr.score(tan_space_covs_val, y_val)
    #print("\nFold", fold+1, "- Acurácia do modelo no conjunto de validação:", accuracy)
    
    y_pred_val = lr.predict(tan_space_covs_val)
    kappa_val  = cohen_kappa_score(y_val, y_pred_val)
    #print("Fold", fold+1, "- Kappa do modelo no conjunto de validação:", kappa_val)
    
    # Armazenar a precisão do modelo no conjunto de validação
    accuracies[fold] = accuracy
    kappas[fold]     = kappa_val

    # Se a precisão do modelo atual for melhor que a anterior, salvar a matriz de projeção
    if kappa_val > best_kappa:
        best_kappa  = kappa_val
        best_c_mean = C_mean

# Mean cross-validation accuracy
mean_accuracy = np.mean(accuracies)
print("\nMean cross-validation accuracy:", mean_accuracy)

# Saving the objects:
with open('C:\\Users\\Laboratorio\\Documents\\GitHub\\real-time-bmi\\best_c_mean.pkl', 'wb') as f: 
    pickle.dump(best_c_mean, f)

# Transform all data to cov matrices
cov_data_train       = Covariances().transform(data)

# Project to euclidean space with best_c_mean
tan_space_covs_train = tangent_space(cov_data_train, best_c_mean)

# Train logistic regression classifier
lr  = LinearDiscriminantAnalysis()
lr.fit(tan_space_covs_train, all_labels)
clf = LogisticRegression().fit(tan_space_covs_train, all_labels)

# Save to file in the current working directory
pkl_filename = "C:\\Users\\Laboratorio\\Documents\\GitHub\\real-time-bmi\\lda.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(clf, file)

plt.figure(figsize=(16,12))
plt.subplot(1,3,1)
plt.imshow(np.mean(np.squeeze(cov_data_train[np.where(all_labels == 1),:,:]), axis = 0))
plt.colorbar(shrink=0.5, aspect=10)
plt.title('Mean cov matrix Right MI')

plt.subplot(1,3,2)
plt.imshow(np.mean(np.squeeze(cov_data_train[np.where(all_labels == 0),:,:]), axis = 0))
plt.colorbar(shrink=0.5, aspect=10)
plt.title('Mean cov matrix Left MI')

plt.subplot(1,3,3)
a = np.mean(np.squeeze(cov_data_train[np.where(all_labels == 0),:,:]), axis = 0) - np.mean(np.squeeze(cov_data_train[np.where(all_labels == 1),:,:]), axis = 0)
plt.imshow(a, vmin = 0, vmax =30)
plt.colorbar(shrink=0.5, aspect=10)
plt.title('Difference cov matrices')

_ = plot_embedding(cov_data_train, all_labels)