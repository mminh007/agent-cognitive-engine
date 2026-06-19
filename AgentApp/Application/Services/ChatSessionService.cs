using System.Linq;
using System.Threading.Tasks;
using Application.Interfaces;
using Domain.Entities;
using Infrastructure.Data;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;

namespace Application.Services
{
    public class ChatSessionService : IChatSessionService
    {
        private readonly AppDbContext _context;
        private readonly int _maxSessionsPerUser;

        public ChatSessionService(AppDbContext context, IConfiguration configuration)
        {
            _context = context;
            // Fetch configuration with a fallback default of 20 sessions
            _maxSessionsPerUser = configuration.GetValue<int>("ChatSettings:MaxSessionsPerUser", 20);
        }

        public async Task<ChatSession> CreateSessionAsync(string userId, string initialTitle)
        {
            // 1. Verify current active sessions for the user
            var userSessions = await _context.ChatSessions
                .Where(s => s.UserId == userId)
                .OrderBy(s => s.CreatedAt)
                .ToListAsync();

            // 2. Pruning logic: Remove oldest sessions if limit is exceeded
            if (userSessions.Count >= _maxSessionsPerUser)
            {
                var sessionsToDeleteCount = userSessions.Count - _maxSessionsPerUser + 1;
                var sessionsToDelete = userSessions.Take(sessionsToDeleteCount);
                _context.ChatSessions.RemoveRange(sessionsToDelete);
            }

            // 3. Initialize and persist the new session
            var newSession = new ChatSession
            {
                UserId = userId,
                Title = initialTitle
            };

            _context.ChatSessions.Add(newSession);
            await _context.SaveChangesAsync();

            return newSession;
        }

        public async Task SaveMessagesAsync(string sessionId, string prompt, string aiResponse)
        {
            var userMsg = new ChatMessage { SessionId = sessionId, Role = "User", Content = prompt };
            var aiMsg = new ChatMessage { SessionId = sessionId, Role = "AI", Content = aiResponse };

            _context.ChatMessages.AddRange(userMsg, aiMsg);
            await _context.SaveChangesAsync();
        }
    }
}