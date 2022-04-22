import socket
import cv2
# from utils.input import RealSense

ip = "127.0.0.1"
port = 5050
print('Connecting to process...')
while True:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((ip, port))
        break
    except socket.error:
        pass
print('Connected to process')

camera = cv2.VideoCapture(0)
# camera = RealSense()
while True:
    _, image = camera.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # rgb, depth = camera.read()
    # image = np.concatenate([cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), depth[..., None]], axis=-1)
    # image = copy.deepcopy(image)
    sock.sendall(image.tobytes())

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

sock.close()
