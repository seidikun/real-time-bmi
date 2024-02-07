import paho.mqtt.client as mqtt
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import threading
import pickle
import configparser
matplotlib.use('TkAgg')  # Defina o backend para TkAgg


sess_filename   = 'D:/Seidi/CG111/RIEMANN_LDA_screening_4_classes_me_CG111_Sess0_Run1'
type_classes    = '4classes'

# Defina as configurações do broker
broker_address  = "localhost" 
port            = 1883  
topic           = "PCA_values"

# Função de callback chamada quando a conexão com o broker é estabelecida
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    client.subscribe(topic)

# Função de callback chamada quando uma mensagem é recebida
def on_message(client, userdata, msg):
    global x, y, label
    data        = msg.payload.decode()
    print(data)
    x, y, label = map(float, data.split(','))
    
# Função para atualizar o scatter plot no thread principal
def update_plot(ax, x, y, label):
    global scatter_plot, classification_text
    if scatter_plot is None:
        scatter_plot = ax.scatter(x, y, color='black', s = 200)
    else:
        scatter_plot.set_offsets(np.c_[x, y])
        
    # Atualiza o texto de classificação
    classification = 'Mão Esquerda' if int(label) == 0 else 'Mão Direita'
    if classification_text is None:
        classification_text = ax.text(0.5, 0.5, classification, ha='center', transform=ax.transAxes, fontsize=30)
    else:
        classification_text.set_text(classification)
        
# Função para rodar o loop MQTT em um thread separado
def mqtt_thread():
    client            = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(broker_address, port, 60)
    client.loop_forever()

scatter_plot        = None
classification_text = None
x,y                 = 0.0, 0.0
label               = 1


with open(sess_filename + '_range_pca.pkl', 'rb') as file:
    x_min, x_max, y_min, y_max = pickle.load(file)

# Cria um gráfico scatter plot vazio
plt.ion()
fig, ax = plt.subplots()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.set_axis_off()  
ax.set_xlim(x_min,x_max)
ax.set_ylim(y_min,y_max)

if type_classes != 'free':
    if type_classes == '2classes':
        nClasses = 2
    elif type_classes == '4classes':
        nClasses = 4
    # Abrir os dados do KDE
    with open(sess_filename + '_kde.pkl', 'rb') as file:
        dict_kde = pickle.load(file)
        
    xx, yy       = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

    colors = ['Blues', 'Reds', 'Oranges', 'Greys']
    for iclass in range(nClasses):
        kde_name   = 'kde' + str(iclass)
        zname      = 'z' + str(iclass)
        z          = dict_kde[kde_name].evaluate(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        plt.contourf(xx, yy, z, alpha=0.5, levels=np.linspace(z.min(), z.max(), 10), cmap=colors[iclass])
    

# Cria e inicia o thread MQTT
mqtt_thread        = threading.Thread(target=mqtt_thread)
mqtt_thread.daemon = True
mqtt_thread.start()

try:
    while True:
        update_plot(ax, x, y, int(label)) 
        plt.pause(0.001) 
except KeyboardInterrupt:
    pass

# Encerra a conexão MQTT quando o programa for encerrado
client.disconnect()

# Aguarde até que a janela do gráfico seja fechada antes de encerrar o programa
plt.ioff()
plt.show()
