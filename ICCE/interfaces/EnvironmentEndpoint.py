import grpc
from ..grpc_interfaces import Environment_pb2, Environment_pb2_grpc

import threading
from concurrent import futures

class EnvironmentServicer(Environment_pb2_grpc.EnvironmentServicer):
    def __init__(self, handshake_cb):
        super().__init__()
        # Store callbacks
        self._on_handshake_and_validate = handshake_cb

    def handshake_and_validate(self, request, context):
        id, status = self._on_handshake_and_validate(request.n_observations, request.n_actions)
        response = Environment_pb2.HandshakeResponse(id=id, status=status)
        return response
    
    def start_simulation(self, request, context):
        print('start_simulation called by ICCE ', request.id)
        response = Environment_pb2.StartResponse(status=1)
        return response
    
    def get_env_data(self, request, context):
        print(f'received from ID: {request.id}')
        response = Environment_pb2.EnvDataResponse()
        response.status = 1
        response.episode = 0
        return response
    
    def set_action_data(self, request, context):
        print('set_action_data called by ICCE ', request.id)
        response = Environment_pb2.ActionResponse(status=1)
        return response
    

class EnvironmentEndpoint():
    def __init__(self, handshake_cb):
        # gRPC server
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        self._server_thread = threading.Thread(target=self.start_server)

        # Environment servicer
        self._servicer = EnvironmentServicer(handshake_cb=handshake_cb)

        Environment_pb2_grpc.add_EnvironmentServicer_to_server(self._servicer, self._server)
        self._server.add_insecure_port('localhost:50051')
    
    def start(self):
        self._server_thread.start()

    def shutdown(self):
        self._server.stop(grace=None)
        self._server_thread.join()

    def start_server(self):
        self._server.start()
        self._server.wait_for_termination()

