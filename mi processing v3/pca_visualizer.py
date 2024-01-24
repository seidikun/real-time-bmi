import paho.mqtt.client as mqtt
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import threading
import pickle
import configparser

matplotlib.use('TkAgg')  # Defina o backend para TkAgg

configParser    = configparser.RawConfigParser()
configFilePath  = r'C:\Users\seidi\Documents\GitHub\real-time-bmi\mi processing v3\config.txt'
configParser.read(configFilePath)
Experiment      = configParser['PARAMETERS']['Experiment']
Participant     = configParser['PARAMETERS']['Participant']
Session_nb      = configParser['PARAMETERS']['Session_nb']
Path_Save       = configParser['PARAMETERS']['Path_Save']
sess_filename   = Path_Save + Participant + '/' + Experiment + '_' + Participant + '_Sess' + Session_nb

sess_filename   = 'C:/Users/seidi/Desktop/Data/EG102/motor_mi_EG102_online_6d'

# Defina as configurações do broker
broker_address = "localhost" 
port           = 1883  
topic          = "PCA_values"

# Abrir os dados do KDE
with open(sess_filename + '_kde0.pkl', 'rb') as file:
    kde0 = pickle.load(file)
with open(sess_filename + '_kde1.pkl', 'rb') as file:
    kde1 = pickle.load(file)

scatter_plot   = None
classification_text = None
x,y            = 0.0, 0.0
label          = 1

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
    print(label)
    global scatter_plot, classification_text
    if scatter_plot is None:
        scatter_plot = ax.scatter(x, y, label='Data Points', color='black', s = 200)
    else:
        scatter_plot.set_offsets(np.c_[x, y])
        
    # Atualiza o texto de classificação
    classification = 'Mão Esquerda' if int(label) == 0 else 'Mão Direita'
    if classification_text is None:
        classification_text = ax.text(0.5, 1.01, classification, ha='center', va='bottom', transform=ax.transAxes, fontsize=24)
    else:
        classification_text.set_text(classification)
        
# Função para rodar o loop MQTT em um thread separado
def mqtt_thread():
    client            = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(broker_address, port, 60)
    client.loop_forever()

# Cria um gráfico scatter plot vazio
plt.ion()
fig, ax = plt.subplots()
ax.set_xlim(-3,3)
ax.set_ylim(-3,3)
ax.set_axis_off()  
x_min, x_max = -3, 3  
y_min, y_max = -3, 3  
xx, yy       = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

# Calcular as densidades no grid
zz0          = kde0(np.vstack([xx.ravel(), yy.ravel()]))
zz1          = kde1(np.vstack([xx.ravel(), yy.ravel()]))

# Plotar as curvas de densidade
contour0     = ax.contourf(xx, yy, zz0.reshape(xx.shape), alpha=0.5, cmap='Blues')
contour1     = ax.contourf(xx, yy, zz1.reshape(xx.shape), alpha=0.5, cmap='Reds')

# Cria e inicia o thread MQTT
mqtt_thread        = threading.Thread(target=mqtt_thread)
mqtt_thread.daemon = True
mqtt_thread.start()

try:
    while True:
        update_plot(ax, x, y, int(label)) 
        plt.pause(0.0001) 
except KeyboardInterrupt:
    pass

# Encerra a conexão MQTT quando o programa for encerrado
client.disconnect()

# Aguarde até que a janela do gráfico seja fechada antes de encerrar o programa
plt.ioff()
plt.show()
