# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import grpc_pb2 as grpc__pb2


class BroadcastServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Communicate = channel.stream_stream(
                '/broadcast.BroadcastService/Communicate',
                request_serializer=grpc__pb2.ClientMessage.SerializeToString,
                response_deserializer=grpc__pb2.ServerMessage.FromString,
                )


class BroadcastServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Communicate(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_BroadcastServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Communicate': grpc.stream_stream_rpc_method_handler(
                    servicer.Communicate,
                    request_deserializer=grpc__pb2.ClientMessage.FromString,
                    response_serializer=grpc__pb2.ServerMessage.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'broadcast.BroadcastService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class BroadcastService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Communicate(request_iterator,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.stream_stream(request_iterator, target, '/broadcast.BroadcastService/Communicate',
            grpc__pb2.ClientMessage.SerializeToString,
            grpc__pb2.ServerMessage.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
