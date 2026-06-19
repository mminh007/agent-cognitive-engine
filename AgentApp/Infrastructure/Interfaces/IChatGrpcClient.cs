using System.Collections.Generic;
using System.Threading;

namespace Infrastructure.Interfaces
{
    public interface IChatGrpcClient
    {
        IAsyncEnumerable<string> StreamChatAsync(string userId, string sessionId, string prompt, CancellationToken cancellationToken);
    }
}