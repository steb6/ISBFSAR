import socket

import cv2
import numpy as np

#Create a socket object.
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

#Bind the socket.
sock.bind(("172.17.224.1", 5051))  # Change ip with ping "$(hostname).local" on wsl

#Start listening.
sock.listen()

#Accept client.
print('Wainting for connections...')
client, addr = sock.accept()
print('Connection received')

image = np.zeros([480, 640, 3], dtype=np.uint8)

#Receive all the bytes and write them into the file.
while True:
    received = client.recv_into(image.data, image.nbytes)

    # print(image)
    cv2.imshow('test', image)
    cv2.waitKey(1)