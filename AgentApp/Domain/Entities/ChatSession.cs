using System;
using System.Collections.Generic;

namespace Domain.Entities
{
    public class ChatSession
    {
        public string Id { get; set; } = Guid.NewGuid().ToString();
        public string UserId { get; set; } = string.Empty;
        public User? User { get; set; }

        public string Title { get; set; } = "New Chat Session";
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
        public DateTime LastUpdated { get; set; } = DateTime.UtcNow;

        // Navigation property
        public ICollection<ChatMessage> Messages { get; set; } = new List<ChatMessage>();
    }
}