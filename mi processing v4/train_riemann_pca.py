import numpy as np
import pandas as pd
from matplotlib             import pyplot as plt
from scipy                  import signal
from scipy.stats            import gaussian_kde
from sklearn.decomposition  import PCA
from sklearn                import svm
from pyriemann.estimation   import Covariances
from pyriemann.utils.mean   import mean_covariance
from pyriemann.tangentspace import tangent_space
import configparser
import pickle
import random


def data_preparation(data_eeg, a, b, columns, dict_inds, dict_labels, Fs, epoch_size, overlap_step):
  # Define parâmetros
  max_trial_t  = 3.75 # tamanho máximo de uma janela de tentativa
  trange        = np.arange(int(Fs*0.5),int(Fs*(0.5 + epoch_size)))

  # Loop pelas janelas
  first_pass = True
  while trange[-1] < int(max_trial_t*Fs):

    for label in dict_labels.keys():
      # Concatena as janelas de ambos os lados
      ts            = [i + trange for i in dict_inds[label]]
      if first_pass:
        data          = data_eeg[ts,:]
        all_labels    = [dict_labels[label] for i in dict_inds[label]]
        all_trial_num = [i for i in range(len(dict_inds[label]))]
      else:
        data          = np.vstack([data, data_eeg[ts,:]])
        all_labels    = np.append(all_labels, [dict_labels[label] for i in dict_inds[label]])
        all_trial_num = np.append(all_trial_num, [i for i in range(len(dict_inds[label]))])

    # Atualiza o índice das janelas
    trange += int(overlap_step*Fs)

    # Atualiza o contador de arquivos
    first_pass = False

  # Aleatoriza as amostras
  p             = np.random.permutation(len(all_labels))
  data          = np.swapaxes(data[p,:,:,],1,2)
  all_labels    = all_labels[p]
  all_trial_num = all_trial_num[p]
  return data, all_labels, all_trial_num

def all_data_epoching(data_eeg, Fs, epoch_size, overlap_step):
  # Define parâmetros
  trange       = np.arange(0,int(Fs*(epoch_size)))
  max_t        = data_eeg.shape[0]
  count_epoch  = 0
  data = []
  # Loop pelas janelas
  while trange[-1] < max_t:
    data.append(data_eeg[trange,:])
    trange      += int(overlap_step*Fs)
    count_epoch += 1
  data = np.array(data)
  data          = np.swapaxes(data,1,2)
  return data


def run_cv_return_c_mean(data, labels, all_trial_num, unique_labels, n_folds = 20):
    # Init variables
    best_acc      = 0
    best_c_mean   = None
    accuracies    = np.zeros(n_folds)
    
    # Cross-validation
    for fold in range(n_folds):
    
        # Randomly choose 2 trial ids for validation
        val_labels       = random.sample(list(unique_labels), 4)
    
        # Trials indexes
        train_inds       = np.where(np.logical_not(np.isin(all_trial_num, val_labels)))[0]
        val_inds         = np.where(np.isin(all_trial_num, val_labels))[0]
    
        # Transform to covariance matrices
        cov_data_train   = Covariances('oas').transform(data[train_inds])
        cov_data_val     = Covariances('oas').transform(data[val_inds])
        
        # Train-validation split
        X_train, y_train = cov_data_train, labels[train_inds]
        X_val, y_val     = cov_data_val,   labels[val_inds]
        
        # Mean covariance matrix
        C_mean           = mean_covariance(X_train)
        
        # Project matrices to tangential space
        tan_space_train  = tangent_space(X_train, C_mean)
        tan_space_val    = tangent_space(X_val, C_mean)
    
        # Treinando o classificador SVC
        classifier       = svm.SVC(kernel='linear', C=1)
        classifier.fit(tan_space_train, y_train)
        
        # Metrics
        print('\nFold', fold+1)
        accuracy         = classifier.score(tan_space_val, y_val)
        print('Accuracy:', accuracy)
        accuracies[fold] = accuracy
    
        # Select best c mean according to kappa
        if accuracy > best_acc:
            best_acc  = accuracy
            best_c_mean = C_mean

    # Calculate mean accuracy
    print("\nMean cross-validation accuracy:", np.mean(accuracies))
    return best_c_mean

def process_data(sess_filename, type_classes):
    
    # ParÂmetros do dado
    Fs           = 512
    b, a         = signal.butter(4, (8, 30), 'bandpass', fs=Fs)
    columns      = ['Channel ' + str(i) for i in  range(1,17)]
    filename     = sess_filename + '.csv'
    df_me        = pd.read_csv(filename, low_memory=False)
    event_id     = pd.to_numeric(df_me['Event Id'], errors='coerce').fillna(0).astype(np.int64)
    epoch_size   = 2   # s
    overlap_step = 0.1 # s
    
    if type_classes == '2classes':
        dict_labels = {
            'left_hand': 0,
            'right_hand': 1
        }
        
        dict_inds = {
            'left_hand':  event_id.index[event_id == 769].tolist(),
            'right_hand': event_id.index[event_id == 770].tolist()
        }
        nClasses = 2
        
    elif type_classes == '4classes':
        dict_labels = {
            'left_hand':  0,
            'right_hand': 1,
            'left_arm':   2,
            'right_arm':  3
        }
        
        dict_inds = {
            'left_hand':  event_id.index[event_id == 769].tolist(),
            'right_hand': event_id.index[event_id == 770].tolist(),
            'left_arm':   event_id.index[event_id == 1089].tolist(),
            'right_arm':  event_id.index[event_id == 1090].tolist()
        }
        nClasses = 4
    
    data_eeg                    = df_me[columns].to_numpy()
    data_eeg                    = signal.filtfilt(b, a, data_eeg.T).T
    
    if type_classes == 'free':
            
        # Janelando o sinal todo
        all_epochs                  = all_data_epoching(data_eeg, Fs, epoch_size, overlap_step)
        cov_data_all                = Covariances(estimator='lwf').transform(all_epochs)
        best_c_mean                 = mean_covariance(cov_data_all)
        tan_vectors_all             = tangent_space(cov_data_all, best_c_mean)
        
        # Aplicando PCA para reduzir para duas dimensões   
        pca                         = PCA(n_components=2)
        pca.fit(tan_vectors_all)
        X_pca                       = pca.transform(tan_vectors_all)
        
        # Calculando a densidade de KDE para cada classe
        x_min, x_max                = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
        y_min, y_max                = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
        xx, yy                      = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        
        plt.scatter(X_pca[:, 0], X_pca[:, 1], s=30, cmap=plt.cm.bwr, edgecolors='k')
        plt.xlabel('PCA Feature 1')
        plt.ylabel('PCA Feature 2')
        plt.title('XPca')
        plt.show()
        
    else: 
        # Janelando as tentativas IM
        data, labels, all_trial_num = data_preparation(data_eeg, a, b, columns, dict_inds, dict_labels,Fs,  epoch_size, overlap_step)
        unique_labels               = np.unique(all_trial_num)
        best_c_mean                 = run_cv_return_c_mean(data, labels, all_trial_num, unique_labels, n_folds = 20)
    
        # Janelando o sinal todo
        all_epochs                  = all_data_epoching(data_eeg, Fs, epoch_size, overlap_step)
        cov_data_all                = Covariances(estimator='lwf').transform(all_epochs)
        tan_vectors_all             = tangent_space(cov_data_all, best_c_mean)
        
        # Aplicando PCA para reduzir para duas dimensões   
        cov_dat                     = Covariances('oas').transform(data)
        X                           = cov_dat
        tan_vectors_mi              = tangent_space(X, best_c_mean)
        pca                         = PCA(n_components=2)
        pca.fit(tan_vectors_mi)
        X_pca                       = pca.transform(tan_vectors_mi)
        
        # Treinando o classificador SVC
        clf                         = svm.SVC(kernel='linear', C=1)
        clf.fit(X_pca, labels)
        
        # Calculando a densidade de KDE para cada classe
        x_min, x_max                = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
        y_min, y_max                = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
        xx, yy                      = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        
        dict_kde = {}
        dict_z   = {}
        colors = ['Blues', 'Reds', 'Oranges', 'Greys']
        for iclass in range(nClasses):
            kde_name           = 'kde' + str(iclass)
            zname              = 'z' + str(iclass)
            dict_kde[kde_name] = gaussian_kde(X_pca[labels == iclass].T)
            dict_z[zname]      = dict_kde[kde_name].evaluate(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            plt.contourf(xx, yy, dict_z[zname], alpha=0.5, levels=np.linspace(dict_z[zname].min(), dict_z[zname].max(), 7), cmap=colors[iclass])
        
        # Desenhando a fronteira de decisão do SVC
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
        
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, s=30, cmap=plt.cm.bwr, edgecolors='k')
        plt.xlabel('PCA Feature 1')
        plt.ylabel('PCA Feature 2')
        plt.title('SVC Decision Boundary with PCA and KDE')
        plt.show()
        
        # Salvando o modelo do classificador e os dados de KDE
        with open(sess_filename + '_classifier.pkl', 'wb') as file:
            pickle.dump(clf, file)
        with open(sess_filename + '_kde.pkl', 'wb') as file:
            pickle.dump(dict_kde, file)
    
    with open(sess_filename + '_range_pca.pkl', 'wb') as f:
        pickle.dump((x_min, x_max, y_min, y_max), f)
        
    with open(sess_filename + '_best_c_mean.pkl', 'wb') as f: 
        pickle.dump(best_c_mean, f)
        
    with open(sess_filename + '_dim_red.pkl', 'wb') as file:
        pickle.dump(pca, file)
        
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        sess_filename = sys.argv[1]
    else:
        print("Please provide a session filename as an argument.")
        sys.exit(1)
    process_data(sess_filename)