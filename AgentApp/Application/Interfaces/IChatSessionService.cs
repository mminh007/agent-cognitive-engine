using System.Threading.Tasks;
using Domain.Entities;

namespace Application.Interfaces
{
    public interface IChatSessionService
    {
        Task<ChatSession> CreateSessionAsync(string userId, string initialTitle);
        Task SaveMessagesAsync(string sessionId, string prompt, string aiResponse);
    }
}