// load_tests/grpc_stream_test.js
import grpc from 'k6/net/grpc';
import { check, sleep } from 'k6';
import { Trend } from 'k6/metrics';

const client = new grpc.Client();
client.load(['.'], 'chat.proto'); // Load the proto file from the current directory

const grpcLatencyTrend = new Trend('custom_grpc_latency', true);

export const options = {
    vus: 20, // Run 20 concurrent gRPC virtual users
    duration: '30s',
};

export default function () {
    client.connect('agent_grpc_server:50051', {
        plaintext: true // Internal Docker network communication without TLS
    });

    const data = {
        user_id: "k6_grpc_user",
        session_id: `grpc_session_${__VU}_${__ITER}`,
        prompt: "Write a Python script to calculate the sum of two numbers."
    };

    // Assumes the service is implemented as a Server Streaming RPC
    const res = client.invoke('chat.AgentService/StreamChat', data);

    check(res, {
        'status is OK': (r) => r && r.status === grpc.StatusOK,
    });

    // Record gRPC request latency
    grpcLatencyTrend.add(res.timers.duration);

    client.close();

    // Simulate user think time between requests
    sleep(1);
}