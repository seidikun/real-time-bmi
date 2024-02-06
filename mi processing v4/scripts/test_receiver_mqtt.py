import paho.mqtt.client as mqtt
import matplotlib
matplotlib.use('TkAgg')  # Defina o backend para TkAgg
import matplotlib.pyplot as plt
import numpy as np
import threading
import pickle

sess_filename =  'C:/Users/seidi/Desktop/Data/EG102/motor_mi_EG102_online_6d'
kde0_filename  = sess_filename + '_kde0.pkl'
kde1_filename  = sess_filename + '_kde1.pkl'

x,y = 0.0, 0.0

# Abrir os dados do KDE
with open(kde0_filename, 'rb') as file:
    kde0 = pickle.load(file)
with open(kde1_filename, 'rb') as file:
    kde1 = pickle.load(file)

# Defina as configurações do broker
broker_address = "localhost"  # Ou o endereço IP do seu broker MQTT
port = 1883  # Porta padrão do MQTT
topic = "PCA_values"

scatter_plot = None

# Função de callback chamada quando a conexão com o broker é estabelecida
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    # Inscreve no tópico
    client.subscribe(topic)

# Função de callback chamada quando uma mensagem é recebida
def on_message(client, userdata, msg):
    global x, y
    data = msg.payload.decode()
    x, y = map(float, data.split(','))
    

# Função para atualizar o scatter plot no thread principal
def update_plot(ax, x, y):
    global scatter_plot
    # Se o scatter plot ainda não foi criado, crie-o
    if scatter_plot is None:
        scatter_plot = ax.scatter(x, y, label='Data Points', color='blue', s = 200)
    else:
        # Atualize os dados do scatter plot existente
        scatter_plot.set_offsets(np.c_[x, y])
        
# Função para rodar o loop MQTT em um thread separado
def mqtt_thread():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(broker_address, port, 60)
    client.loop_forever()

# Cria um gráfico scatter plot vazio
plt.ion()
fig, ax = plt.subplots()
# fig.add_axes([0, 0, 1, 1]) 
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
mqtt_thread  = threading.Thread(target=mqtt_thread)
mqtt_thread.daemon = True
mqtt_thread.start()

try:
    while True:
        update_plot(ax, x, y) 
        plt.pause(0.0001) 
except KeyboardInterrupt:
    pass

# Encerra a conexão MQTT quando o programa for encerrado
client.disconnect()

# Aguarde até que a janela do gráfico seja fechada antes de encerrar o programa
plt.ioff()
plt.show()
