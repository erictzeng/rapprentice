import cPickle
import zmq

class Request(object):
    def __init__(self, id_str, payload=None):
        self.id_str = id_str
        self.payload = payload

    def serialize(self):
        return cPickle.dumps((self.id_str, self.payload))

    @staticmethod
    def Deserialize(data):
        id_str, payload = cPickle.loads(data)
        return Request(id_str, payload)

class Client(object):
    def __init__(self, url='tcp://localhost:5555'):
        self.url = url
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(url)

    def send_request(self, req):
        assert isinstance(req, Request)
        self.socket.send(req.serialize())
        reply = self.socket.recv()
        return cPickle.loads(reply)

    def lookup(self, demo_i, demo_j):
        return self.send_request(Request('lookup', (demo_i, demo_j)))

    def fetch_mat_shape(self):
        return self.send_request(Request('fetch_mat_shape'))


class Server(object):
    def __init__(self, func_mat_file='/media/3tb/demos/func_matrix.pkl'):
        print 'loading', func_mat_file, '...'
        with open(func_mat_file, 'r') as f:
            self.func_matrix = cPickle.load(f)
        print 'done'
        
        self.context = zmq.Context()
        self.socket = zmq.Socket(self.context, zmq.REP)
        self.socket.bind("tcp://*:5555")

    def process_request(self, req):
        assert isinstance(req, Request)
        reply = None
        if req.id_str == 'lookup':
            demo_i, demo_j = req.payload
            print '>>> processing request: lookup %d -> %d' % (demo_i, demo_j)
            reply = self.func_matrix[demo_i][demo_j]
        elif req.id_str == 'fetch_mat_shape':
            print '>>> processing request: fetch_mat_shape'
            reply = (len(self.func_matrix), len(self.func_matrix[0]))
        else:
            print '>>> ERROR: unknown request: %s, payload: %s' % (req.id_str, repr(req.payload))
        return reply

    def loop(self):
        while True:
            data = self.socket.recv()
            req = Request.Deserialize(data)
            reply = None
            try:
                reply = self.process_request(req)
            except:
                import traceback
                traceback.print_exc()
                reply = None
            self.socket.send(cPickle.dumps(reply))

def main():
    Server().loop()

if __name__ == '__main__':
    main()
