import train_riemann_pca as triem
path_data = 'C:/Users/seidi/Desktop/Data/TEST/'

filename = path_data + 'motor_mi_EG102_online_6d'

triem.process_data(filename, '2classes')