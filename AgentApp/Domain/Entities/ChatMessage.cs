using System;

namespace Domain.Entities
{
    public class ChatMessage
    {
        public string Id { get; set; } = Guid.NewGuid().ToString();
        public string SessionId { get; set; } = string.Empty;
        public ChatSession? Session { get; set; }

        public string Role { get; set; } = string.Empty; // "User" or "AI"
        public string Content { get; set; } = string.Empty;
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    }
}