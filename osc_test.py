from pythonosc import udp_client
import time

client = udp_client.SimpleUDPClient("192.168.1.106", 9000)  # TD PC の IP

while True:
    client.send_message("/touch", [1, 0.5, -0.3])
    time.sleep(0.1)
    client.send_message("/touch", [0, 0.5, -0.3])
    time.sleep(1)
