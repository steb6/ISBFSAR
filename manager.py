from multiprocessing.managers import BaseManager
from collections import defaultdict
from queue import Queue


queues = defaultdict(lambda: Queue(1))


if __name__ == '__main__':

    BaseManager.register('get_queue', callable=lambda name: queues[name])

    m = BaseManager(address=('localhost', 50000), authkey=b'abracadabra')
    s = m.get_server()
    s.serve_forever()
