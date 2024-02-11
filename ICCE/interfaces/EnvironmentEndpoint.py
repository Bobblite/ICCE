import grpc
from ..grpc_interfaces import Environment_pb2, Environment_pb2_grpc

import threading
from concurrent import futures

class EnvironmentServicer(Environment_pb2_grpc.EnvironmentServicer):
    def handshake_and_validate(self, request, context):
        print('handshake_and_validate called by ICCE ', request.id)
        response = Environment_pb2.HandshakeResponse(status=1)
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
    def __init__(self):
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        self._server_thread = threading.Thread(target=self.start_server)

        Environment_pb2_grpc.add_EnvironmentServicer_to_server(EnvironmentServicer(), self._server)
        self._server.add_insecure_port('localhost:50051')
    
    def start(self):
        self._server_thread.start()

    def shutdown(self):
        self._server.stop(grace=None)
        self._server_thread.join()

    def start_server(self):
        self._server.start()
        self._server.wait_for_termination()

