using AgentApp.Protos; // Generated from protobuf
using Grpc.Core;
using Grpc.Net.Client;
using Infrastructure.Interfaces;
using Microsoft.Extensions.Configuration;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Threading;

namespace Infrastructure.Services
{
    public class ChatGrpcClient : IChatGrpcClient
    {
        private readonly AgentService.AgentServiceClient _grpcClient;

        public ChatGrpcClient(IConfiguration configuration)
        {
            // Fetch gRPC server URL from configuration instead of hardcoding
            var grpcUrl = configuration["GrpcSettings:CoreEngineUrl"] ?? "http://localhost:50051";
            var channel = GrpcChannel.ForAddress(grpcUrl);
            _grpcClient = new AgentService.AgentServiceClient(channel);
        }

        public async IAsyncEnumerable<string> StreamChatAsync(
            string userId,
            string sessionId,
            string prompt,
            [EnumeratorCancellation] CancellationToken cancellationToken)
        {
            // 1. Initialize protobuf request payload
            var request = new ChatRequest
            {
                UserId = userId,
                SessionId = sessionId,
                Prompt = prompt
            };

            // 2. Invoke RPC Streaming call to Python Core
            using var streamingCall = _grpcClient.StreamChat(request, cancellationToken: cancellationToken);

            // 3. Yield each chunk as it arrives from the stream
            await foreach (var response in streamingCall.ResponseStream.ReadAllAsync(cancellationToken))
            {
                yield return response.Chunk;
            }
        }
    }
}