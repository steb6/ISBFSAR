import socket

import cv2
import numpy as np

#Connect to the server.
print('Connecting...')
out_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
out_sock.connect(("172.17.224.1", 5051))
print('Connected to sink')

#Create a socket object.
in_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

#Bind the socket.
in_sock.bind(("127.0.0.1", 5050))

#Start listening.
in_sock.listen()

#Accept client.
print('Waiting for connections...')
client, addr = in_sock.accept()
print('Connection received')

image = np.zeros([480, 640, 3], dtype=np.uint8)

#Receive all the bytes and write them into the file.
while True:
    received = client.recv_into(image.data, image.nbytes)
    # print(image)
    out_sock.sendall(image.tobytes())