import paho.mqtt.client as mqtt
import json, math,time
from bhaptics import better_haptic_player as player
from time import sleep

class VESTMQTT(mqtt.Client):
  
    def connectVEST(self):
        self.connect('10.1.1.243', 1883, 600)
        player.initialize()
        player.register("lfoot", "C:\Users\seidi\Desktop\real-time-bmi-main\BMI VR Demo\\left_foot.tact")
        player.register("rfoot", "C:\Users\seidi\Desktop\real-time-bmi-main\BMI VR Demo\\right_foot.tact")
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
            if msgDict['index'] == 2:
                player.submit_registered("lfoot")
            elif msgDict['index'] == 3:
                player.submit_registered("rfoot")

            print(msgDict)

vest = VESTMQTT()
vest.connectVEST()
