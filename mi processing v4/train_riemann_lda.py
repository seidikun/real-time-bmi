import numpy as np
import pandas as pd
from matplotlib                    import pyplot as plt
from scipy                         import signal
from scipy.stats                   import gaussian_kde
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn                       import svm
from pyriemann.estimation          import Covariances
from pyriemann.clustering          import Potato
from pyriemann.utils.mean          import mean_covariance
from pyriemann.tangentspace        import tangent_space
from functools                     import partial
import pickle
import random

def data_preparation(data_eeg, dict_classes, list_classes, Fs, epoch_size, overlap_step):
    # Define parâmetros
    max_trial_t = 3.75  # tamanho máximo de uma janela de tentativa
    trange = np.arange(int(Fs*0.5), int(Fs*(0.5 + epoch_size)))
    baseline_start = 5 * Fs  # Início da baseline em segundos convertidos para amostras
    baseline_end   = 25 * Fs  # Fim da baseline em segundos convertidos para amostras

    # Loop pelas janelas
    first_pass = True
    while trange[-1] < int(max_trial_t*Fs):
        for label in list_classes:
            if label == 'baseline':
                # Para 'baseline', calcula os timestamps dentro do intervalo de interesse
                ts = [i + trange for i in range(baseline_start, baseline_end, int(Fs)) if i + trange[-1] <= baseline_end]
            else:
                # Para os outros labels, usa os timestamps definidos em dict_classes
                ts = [i + trange for i in dict_classes[label][1]]

            if first_pass:
                data = data_eeg[ts, :]
                if label == 'baseline':
                    all_labels = [dict_classes[label][0] for _ in range(len(ts))]
                else:
                    all_labels = [dict_classes[label][0] for i in dict_classes[label][1]]
                all_trial_num = [i for i in range(len(ts))]
                first_pass = False
            else:
                data = np.vstack([data, data_eeg[ts, :]])
                if label == 'baseline':
                    all_labels = np.append(all_labels, [dict_classes[label][0] for _ in range(len(ts))])
                else:
                    all_labels = np.append(all_labels, [dict_classes[label][0] for i in dict_classes[label][1]])
                all_trial_num = np.append(all_trial_num, [i for i in range(len(ts))])
        trange += int(overlap_step*Fs)
    data = np.swapaxes(data, 1, 2)
    return data, all_labels, all_trial_num


def all_data_epoching(data_eeg, Fs, epoch_size, overlap_step):
  trange       = np.arange(0,int(Fs*(epoch_size)))
  max_t        = data_eeg.shape[0]
  data         = []
  while trange[-1] < max_t:
    data.append(data_eeg[trange,:])
    trange      += int(overlap_step*Fs)
  return np.swapaxes(np.array(data),1,2)

def run_cv_return_c_mean(cov_data, labels, all_trial_num, unique_labels, dict_classes, list_classes, n_folds = 10):
    best_acc      = 0
    best_c_mean   = None
    accuracies    = np.zeros(n_folds)
    
    # Cross-validation
    for fold in range(n_folds):
        # Randomly choose 2 trial ids for validation and the rest for training
        val_labels       = random.sample(list(unique_labels), 2)
        val_inds         = np.where(np.isin(all_trial_num, val_labels))[0]
        train_inds       = np.where(np.logical_not(np.isin(all_trial_num, val_labels)))[0]
        
        # Train-validation split
        X_train, y_train, X_val, y_val = cov_data[train_inds,:,:], labels[train_inds], cov_data[val_inds, :, :], labels[val_inds]
        
        # Mean covariance matrix
        C_mean           = mean_covariance(X_train)
        
        # Project matrices to tangential space
        tan_space_train  = tangent_space(X_train, C_mean)
        tan_space_val    = tangent_space(X_val, C_mean)
                
        lda              = LDA(n_components=2).fit(tan_space_train, y_train)
        X_lda            = lda.transform(tan_space_train)
        X_lda_val        = lda.transform(tan_space_val)
        
        # Treinando o classificador SVC
        classifier       = svm.SVC(kernel='rbf', C=1)
        classifier.fit(X_lda, y_train)
        
        # Metrics
        print('\nFold', fold+1)
        accuracy         = classifier.score(X_lda_val, y_val)
        print('Accuracy:', accuracy)
        accuracies[fold] = accuracy
        
        x_min, x_max, y_min, y_max = np.min(X_lda[:, 0])-1, np.max(X_lda[:, 0])+1, np.min(X_lda[:, 1])-1, np.max(X_lda[:, 1])+1
        xx, yy                     = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        
        # Treinando o classificador SVC
        clf                         = svm.SVC(kernel='linear', C=1)
        clf.fit(X_lda, y_train)
        
        plt.figure(figsize = (10, 10))
        dict_kde, dict_z            = {}, {}
        colors_kde = ['Greys', 'Blues', 'Reds', 'Oranges', 'Purples']
        # colors_sca = ['grey', 'blue', 'red', 'orange', 'purple']
        for label in list_classes:
            kde_name           = 'kde_' + label
            zname              = 'z_' + label
            ind_class          = y_train == dict_classes[label][0]
            iclass             = dict_classes[label][0]
            dict_kde[kde_name] = gaussian_kde(X_lda[ind_class].T)
            dict_z[zname]      = dict_kde[kde_name].evaluate(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            plt.contourf(xx, yy, dict_z[zname], alpha=0.5, levels=np.linspace(dict_z[zname].min(), dict_z[zname].max(), 10), cmap=colors_kde[iclass])
            # plt.scatter(X_lda[ind_class, 0], X_lda[ind_class, 1], s=30, color=colors_sca[iclass], edgecolors='k', label = label)


        for iclass in range(len(list_classes)):
            ind_class          = y_train == iclass
            
        plt.legend()
        # Desenhando a fronteira de decisão do SVC
        Z                           = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
        plt.xlabel('LDA Feature 1')
        plt.ylabel('LDA Feature 2')
        plt.title('SVC Decision Boundary with LDA and KDE')
        plt.show()
        
        # Select best c mean according to kappa
        if accuracy > best_acc:
            best_acc  = accuracy
            best_c_mean = C_mean

    print("\nMean cross-validation accuracy:", np.mean(accuracies))
    return best_c_mean

@partial(np.vectorize, excluded=['potato'])
def get_zscores(cov_00, cov_01, cov_11, potato):
    cov = np.array([[cov_00, cov_01], [cov_01, cov_11]])
    with np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        return potato.transform(cov[np.newaxis, ...])


def process_data(sess_filename, type_classes):
    # Parâmetros do dado
    Fs           = 512
    b, a         = signal.butter(4, (8, 30), 'bandpass', fs=Fs)
    columns      = ['Channel ' + str(i) for i in  range(1,17)]
    filename     = sess_filename + '.csv'
    df_me        = pd.read_csv(filename, low_memory=False)
    data_eeg     = signal.filtfilt(b, a, df_me[columns].to_numpy().T).T
    event_id     = pd.to_numeric(df_me['Event Id'], errors='coerce').fillna(0).astype(np.int64)
    epoch_size   = 1   # s
    overlap_step = 0.0625 # s
    dict_classes = {
        'baseline':   (0,0),
        'left_hand':  (1,event_id.index[event_id == 769].tolist()),
        'right_hand': (2,event_id.index[event_id == 770].tolist()),
        'left_elbow': (3,event_id.index[event_id == 1089].tolist()),
        'right_elbow':(4, event_id.index[event_id == 1090].tolist())
    }    
    z_th        = 3.0 # limiar usando z-score
    
    if type_classes == '2classes':
        list_classes = ['baseline', 'left_hand', 'right_hand']
        nClasses     = 3
    elif type_classes == '4classes':
        # list_classes = ['left_hand', 'right_hand', 'left_arm', 'right_arm']
        list_classes = ['left_hand', 'right_hand', 'left_elbow', 'right_elbow']
        nClasses = 4
    
    # Janelando o sinal todo
    all_epochs                 = all_data_epoching(data_eeg, Fs, epoch_size, overlap_step)
    cov_data_all               = Covariances(estimator='oas').transform(all_epochs)

    # Janelando as tentativas IM
    mi_epochs, labels, trials  = data_preparation(data_eeg, dict_classes ,list_classes, Fs, epoch_size, overlap_step)
    unique_labels              = np.unique(trials)
    cov_data_mi                = Covariances('oas').transform(mi_epochs)
    
    rpotato                    = Potato(metric='riemann', threshold=z_th).fit(cov_data_all)
    rp_center                  = rpotato._mdm.covmeans_[0]
    rp_labels                  = rpotato.predict(cov_data_mi)
    inds_potato     = np.where(rp_labels == 1)[0]
    inds_outlier    = np.where(rp_labels == 0)[0]
    print(str(len(inds_outlier)/len(inds_potato)*100) + '% of epochs discarded by potato')
    cov_data_mi, labels, trials =  cov_data_mi[inds_potato,:,:], labels[inds_potato], trials[inds_potato]
    
    best_c_mean                = run_cv_return_c_mean(cov_data_mi, labels, trials, unique_labels, dict_classes, list_classes, n_folds = 10)
    
    # Aplicando LDA para reduzir para duas dimensões   
    tan_vectors_mi             = tangent_space(cov_data_mi, best_c_mean)
    lda                        = LDA(n_components=2)
    lda.fit(tan_vectors_mi, labels)
    X_lda                      = lda.transform(tan_vectors_mi)
    x_min, x_max, y_min, y_max = np.min(X_lda[:, 0])-1, np.max(X_lda[:, 0])+1, np.min(X_lda[:, 1])-1, np.max(X_lda[:, 1])+1
    xx, yy                     = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    # Treinando o classificador SVC
    clf                         = svm.SVC(kernel='linear', C=1)
    clf.fit(X_lda, labels)
    
    plt.figure(figsize = (10, 10))
    dict_kde, dict_z            = {}, {}
    colors_kde = ['Greys', 'Blues', 'Reds', 'Oranges', 'Purples']
    for label in list_classes:
        kde_name           = 'kde_' + label
        zname              = 'z_' + label
        ind_class          = labels == dict_classes[label][0]
        iclass             = dict_classes[label][0]
        dict_kde[kde_name] = gaussian_kde(X_lda[ind_class].T)
        dict_z[zname]      = dict_kde[kde_name].evaluate(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        plt.contourf(xx, yy, dict_z[zname], alpha=0.5, levels=np.linspace(dict_z[zname].min(), dict_z[zname].max(), 10), cmap=colors_kde[iclass])
    
    
    
    # Desenhando a fronteira de decisão do SVC
    Z                           = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    
    # Desenhando pontos
    plt.scatter(X_lda[:, 0], X_lda[:, 1], c=labels, s=30, cmap=plt.cm.bwr, edgecolors='k')
    plt.xlabel('LDA Feature 1')
    plt.ylabel('LDA Feature 2')
    plt.title('SVC Decision Boundary with LDA and KDE')
    plt.show()
    
    # Salvando variáveis
    with open(sess_filename + '_classifier.pkl', 'wb') as file:
        pickle.dump(clf, file)
    with open(sess_filename + '_kde.pkl', 'wb') as file:
        pickle.dump(dict_kde, file)
    with open(sess_filename + '_range_red_dim.pkl', 'wb') as f:
        pickle.dump((x_min, x_max, y_min, y_max), f)
    with open(sess_filename + '_best_c_mean.pkl', 'wb') as f: 
        pickle.dump(best_c_mean, f)
    with open(sess_filename + '_dim_red.pkl', 'wb') as file:
        pickle.dump(lda, file)
        
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        sess_filename = sys.argv[1]
    else:
        # print("Please provide a session filename as an argument.")
        # sys.exit(1)
        sess_filename = 'C:/Users/seidi/Desktop/Data/TEST/RIEMANN_LDA_screening_2_classes_mi_TEST_Sess0000_Run1'
        process_data(sess_filename, '2classes')
    # process_data(sess_filename)