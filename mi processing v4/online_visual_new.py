import paho.mqtt.client as mqtt
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import threading
import pickle

# Certifique-se de usar um backend que seja compatível com a maneira como você deseja exibir os gráficos.
matplotlib.use('TkAgg')

def run_visualization(sess_filename):
    # Configurações do broker MQTT
    broker_address  = "localhost"
    port            = 1883
    topic           = "PCA_values"

    global scatter_plot, classification_text, x, y, label
    scatter_plot        = None
    classification_text = None
    x, y                = 0.0, 0.0
    label               = 1

    # Funções de callback para o cliente MQTT
    def on_connect(client, userdata, flags, rc):
        print("Connected with result code "+str(rc))
        client.subscribe(topic)

    def on_message(client, userdata, msg):
        global x, y, label
        data        = msg.payload.decode()
        print(data)
        x, y, label = map(float, data.split(','))

    # Função para atualizar o gráfico
    def update_plot(ax, x, y, label):
        global scatter_plot, classification_text
        if scatter_plot is None:
            scatter_plot = ax.scatter(x, y, color='black', s=200)
        else:
            scatter_plot.set_offsets(np.c_[x, y])
        
        # Atualiza o texto de classificação
        classification = 'Mão Esquerda' if int(label) == 0 else 'Mão Direita'
        if classification_text is None:
            classification_text = ax.text(0.5, 0.5, classification, ha='center', transform=ax.transAxes, fontsize=30)
        else:
            classification_text.set_text(classification)

    # Função para executar o loop MQTT em um thread separado
    def mqtt_thread_function():
        client = mqtt.Client()
        client.on_connect = on_connect
        client.on_message = on_message
        client.connect(broker_address, port, 60)
        client.loop_forever()

    # Carrega os limites do PCA
    with open(sess_filename + '_range_pca.pkl', 'rb') as file:
        x_min, x_max, y_min, y_max = pickle.load(file)

    # Cria um gráfico scatter plot vazio
    plt.ion()
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_axis_off()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Inicia o thread MQTT
    mqtt_thread = threading.Thread(target=mqtt_thread_function)
    mqtt_thread.daemon = True
    mqtt_thread.start()

    try:
        while True:
            update_plot(ax, x, y, int(label))
            plt.pause(0.001)
    except KeyboardInterrupt:
        pass

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    # Este bloco é apenas para execução direta deste script e não será usado ao chamar run_visualization de outro script.
    sess_filename = "D:/Seidi/TEST/RIEMANN_LDA_screening_2_classes_mi_TEST_Sess0000_Run1"
    run_visualization(sess_filename)
