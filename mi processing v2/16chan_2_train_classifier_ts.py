# -*- coding: utf-8 -*-

import pickle
import random
import numpy  as np
import pandas as pd

from scipy                         import signal
from sklearn.metrics               import cohen_kappa_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model          import LogisticRegression
from pyriemann.estimation          import Covariances
from pyriemann.tangentspace        import tangent_space
from pyriemann.utils.mean          import mean_covariance

import configparser 

configParser = configparser.RawConfigParser()   
configFilePath = r'.\config.txt'
configParser.read(configFilePath)

Experiment      = configParser['PARAMETERS']['Experiment']
Participant     = configParser['PARAMETERS']['Participant']
Session_nb      = configParser['PARAMETERS']['Session_nb']
Path_Project    = configParser['PARAMETERS']['Path_Project']
Path_Save       = configParser['PARAMETERS']['Path_Save']
sess_filename   = Path_Save + Participant + '/' + Experiment + '_' + Participant + '_Sess' + Session_nb
filename        = sess_filename + '.csv'
pkl_filename    = sess_filename + '_lda.pkl'
c_mean_filename = sess_filename + '_best_c_mean.pkl'

#################### TROQUE O NOME DO ARQUIVO AQUI ###########################

# Parameters
overlap_step = 0.1  # tamanho do passo para sobreposição de janelas
max_trial_t  = 3.75 # tamanho máximo de uma janela de tentativa
b, a         = signal.butter(4, (8,30), 'bandpass', fs=512)

# Read .csv file
df_me        = pd.read_csv(filename)

# Get indexes of left MI and right MI
event_id     = pd.to_numeric(df_me['Event Id'], errors='coerce').fillna(0).astype(np.int64)
left_inds    = event_id.index[event_id == 769].tolist()
right_inds   = event_id.index[event_id == 770].tolist()
  
# Sample frequency
Fs           = 1/np.mean(np.diff(df_me['Time:512Hz']))

# Get EEG data only and band-pass filter
data_eeg     = df_me.iloc[:, 2:18].values
data_eeg     = signal.filtfilt(b, a, data_eeg.T).T
    
# Time vector of epoch
trange       = np.arange(int(Fs*0.5),int(Fs*2.5), 4) # O 4 é pra realizar um downsample por 4
  
# Windowing
while trange[-1] < int(max_trial_t*Fs):        

  # Left MI epochs
  ts            = [i + trange for i in left_inds]
  epochs_left   = data_eeg[ts,:]
  labels        = [0 for i in left_inds]
  trial_num     = [i for i in range(len(left_inds))]

  # Right MI epochs
  ts            = [i + trange for i in right_inds]
  epochs_right  = data_eeg[ts,:]
  epochs_mne    = np.vstack([epochs_left, epochs_right])
  labels        = np.append(labels, [1 for i in right_inds])
  trial_num     = np.append(trial_num, [i for i in range(len(right_inds))])

  # Create epochs stack, labels array and trial id array
  if 'data' not in locals():
    data          = epochs_mne
    all_labels    = labels
    all_trial_num = trial_num
  else:
    data          = np.vstack([data, epochs_mne])
    all_labels    = np.append(all_labels, labels)
    all_trial_num = np.append(all_trial_num, trial_num)

  # Slide window
  trange += int(overlap_step*Fs)

# Randomization
p             = np.random.permutation(len(all_labels))
data          = np.swapaxes(data[p,:,:,],1,2)
all_labels    = all_labels[p]
all_trial_num = all_trial_num[p]


# Inicializar variáveis
best_accuracy = 0
best_kappa    = 0
best_c_mean   = None
n_folds       = 20

# Criar lista com os 20 rótulos únicos de trial
unique_labels = np.unique(all_trial_num)

# Inicializar array para armazenar resultados da validação cruzada
accuracies    = np.zeros(n_folds)
kappas        = np.zeros(n_folds)

# Fazer a validação cruzada
for fold in range(n_folds):

    # Escolher aleatoriamente 2 rótulos para o conjunto de validação
    val_labels = random.sample(list(unique_labels), 2)

    # Selecionar os índices dos dados de treinamento e validação
    train_inds           = np.where(np.logical_not(np.isin(all_trial_num, val_labels)))[0]
    val_inds             = np.where(np.isin(all_trial_num, val_labels))[0]

    # Transformar os dados em matrizes de covariância
    cov_data_train       = Covariances().transform(data[train_inds])
    cov_data_val         = Covariances().transform(data[val_inds])
    
    # Dividir os dados em conjunto de treinamento, validação e teste
    X_train, y_train     = cov_data_train, all_labels[train_inds]
    X_val, y_val         = cov_data_val, all_labels[val_inds]
    
    # Calcular a média das matrizes de covariância do conjunto de treinamento
    C_mean               = mean_covariance(X_train)
    
    # Projetar as matrizes de covariância no plano tangencial
    tan_space_covs_train = tangent_space(X_train, C_mean)
    tan_space_covs_val   = tangent_space(X_val, C_mean)
    
    # Treinar o modelo de LDA no conjunto de treinamento
    lda = LinearDiscriminantAnalysis()
    lda = LogisticRegression(random_state=0)

    lda.fit(tan_space_covs_train, y_train)
    
    # Avaliar a precisão do modelo no conjunto de validação
    accuracy = lda.score(tan_space_covs_val, y_val)
    print("\nFold", fold+1, "- Acurácia do modelo no conjunto de validação:", accuracy)
    
    y_pred_val = lda.predict(tan_space_covs_val)
    kappa_val  = cohen_kappa_score(y_val, y_pred_val)
    print("Fold", fold+1, "- Kappa do modelo no conjunto de validação:", kappa_val)
    
    # Armazenar a precisão do modelo no conjunto de validação
    accuracies[fold] = accuracy
    kappas[fold]     = kappa_val

    # Se a precisão do modelo atual for melhor que a anterior, salvar a matriz de projeção
    if kappa_val > best_kappa:
        best_kappa = kappa_val
        best_c_mean   = C_mean

# Calcular a precisão média do modelo na validação cruzada
mean_accuracy = np.mean(accuracies)
print("\nPrecisão média do modelo na validação cruzada:", mean_accuracy)

# Saving the objects:
with open(c_mean_filename, 'wb') as f: 
    pickle.dump(best_c_mean, f)

# Transformar os dados em matrizes de covariância
cov_data_train       = Covariances().transform(data)

# Calcular a média das matrizes de covariância do conjunto de treinamento
C_mean               = mean_covariance(cov_data_train)

# Projetar as matrizes de covariância no plano tangencial
tan_space_covs_train = tangent_space(cov_data_train, C_mean)

# Treinar o modelo de LDA no conjunto de treinamento
lda = LinearDiscriminantAnalysis()
lda.fit(tan_space_covs_train, all_labels)

clf = LogisticRegression().fit(tan_space_covs_train, all_labels)

# Save to file in the current working directory
with open(pkl_filename, 'wb') as file:
    pickle.dump(clf, file)

