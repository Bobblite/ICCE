import grpc
from ..grpc_interfaces import Environment_pb2, Environment_pb2_grpc

from concurrent import futures

class EnvironmentServicer(Environment_pb2_grpc.EnvironmentServicer):
    def GetEnvData(self, request, context):
        print(f'received from ID: {request.id}')
        response = Environment_pb2.EnvData()
        response.status = 1
        response.episode = 0
        return response

class EnvironmentEndpoint():
    def __init__(self):
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        Environment_pb2_grpc.add_EnvironmentServicer_to_server(EnvironmentServicer(), self._server)
        self._server.add_insecure_port('localhost:50051')
    
    def start(self):
        self._server.start()
        self._server.wait_for_termination()
