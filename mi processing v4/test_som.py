import numpy as np
import pyaudio
import threading
from threading import Thread
import time
import paho.mqtt.client as mqtt
import configparser
import pickle
import time
import math

def elliptical_points(a, b, h, k, angle_step, duration):
    """Gera pontos em uma elipse por um tempo determinado."""
    start_time = time.time()
    angle = 0
    while time.time() - start_time < duration:
        x = a * math.cos(angle) + h
        y = b * math.sin(angle) + k
        yield x, y
        angle += angle_step
        if angle >= 2 * math.pi:
            angle -= 2 * math.pi

def on_connect(client, userdata, flags, rc):
    """Callback para quando o cliente se conecta ao broker."""
    if rc == 0:
        print("Connected to MQTT Broker!")
    else:
        print("Failed to connect, return code %d\n", rc)

configParser    = configparser.RawConfigParser()
configFilePath  = r'C:/Users/seidi/Documents/GitHub/real-time-bmi/mi processing v3/config.txt'
configParser.read(configFilePath)
Experiment      = configParser['PARAMETERS']['Experiment']
type_classes    = configParser['PARAMETERS']['type_classes']
Participant     = configParser['PARAMETERS']['Participant']
Session_nb      = configParser['PARAMETERS']['Session_nb']
Path_Save       = configParser['PARAMETERS']['Path_Save']
sess_filename   = Path_Save + Participant + '/' + Experiment + '_' + type_classes + '_' + Participant + '_Sess' + Session_nb


# Parâmetros da elipse
with open(sess_filename + '_range_pca.pkl', 'rb') as file:
    x_min, x_max, y_min, y_max = pickle.load(file)
a = (x_max - x_min) / 2
b = (y_max - y_min) / 2
h = x_min + a
k = y_min + b

# Parâmetros MQTT
broker_address = "localhost"
port = 1883
topic = "PCA_values"

# Configura o cliente MQTT
client = mqtt.Client("EllipsePublisher")
client.on_connect = on_connect
client.connect(broker_address, port=port)

# Inicia o loop MQTT
client.loop_start()

try:
    # A duração é definida aqui como 10 segundos
    duration = 100
    for x, y in elliptical_points(a, b, h, k, 2 * math.pi / 512, duration):
        message = str(x) + ',' + str(y) + ',1' 
        client.publish(topic, message)
        time.sleep(1/128)  # Aguarda para atingir 512Hz
        print(x,y)
except KeyboardInterrupt:
    print("Encerrando simulação.")

client.loop_stop()
client.disconnect()