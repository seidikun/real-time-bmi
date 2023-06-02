import paho.mqtt.client as mqtt
import json, math,time
from bhaptics import better_haptic_player as player
from time import sleep

class VESTMQTT(mqtt.Client):
  
    def connectVEST(self):
        self.connect('10.1.1.243', 1883, 600)
        player.initialize()
        player.register("full_body", "C:\\Users\\Laboratorio\\Desktop\\bHapticsMqtt-main\\full_body_quick_vibration.tact")
        player.register("lhit", "C:\\Users\\Laboratorio\\Desktop\\bHapticsMqtt-main\\left_hit.tact")
        player.register("rhit", "C:\\Users\\Laboratorio\\Desktop\\bHapticsMqtt-main\\right_hit.tact")
        player.register("heart", "C:\\Users\\Laboratorio\\Desktop\\bHapticsMqtt-main\\heart_thump_one_time.tact")
        player.register("dash", "C:\\Users\\Laboratorio\\Desktop\\bHapticsMqtt-main\\dash.tact")
        player.register("grass","C:\\Users\Laboratorio\\Desktop\\bHapticsMqtt-main\\GrassSens.tact")

        rc = 0
        while rc == 0:
            rc = self.loop()
        return rc

    def on_connect(self, mqttc, obj, flags, rc):
        print("Connected with result code "+str(rc))
        self.subscribe('tactic_suit')
    

    def on_message(self, mqttc, obj, msg):
        if msg.topic == 'tactic_suit':
            msgDict = json.loads(msg.payload)
            if msgDict['index'] == 1:
                player.submit_registered("full_body")
            elif msgDict['index'] == 2:
                player.submit_registered("lhit")
            elif msgDict['index'] == 3:
                player.submit_registered("rhit")
            elif msgDict['index'] == 4:
                player.submit_registered("heart")
            elif msgDict['index'] == 5:
                player.submit_registered("dash")
            elif msgDict['index'] == 6:
                player.submit_registered("grass")

            print(msgDict)

vest = VESTMQTT()
vest.connectVEST()
