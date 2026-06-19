using System;
using System.Security.Claims;
using System.Threading;
using System.Threading.Tasks;
using Application.Interfaces;
using Infrastructure.Interfaces;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;

namespace AgentApp.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class ChatController : ControllerBase
    {
        private readonly IChatGrpcClient _chatGrpcClient;
        private readonly IChatSessionService _chatSessionService;

        public ChatController(IChatGrpcClient chatGrpcClient, IChatSessionService chatSessionService)
        {
            _chatGrpcClient = chatGrpcClient;
            _chatSessionService = chatSessionService;
        }

        [HttpPost("stream")]
        public async Task StreamChat([FromForm] string prompt, CancellationToken cancellationToken)
        {
            Response.ContentType = "text/event-stream";

            // Check authentication status
            bool isAuthenticated = User.Identity?.IsAuthenticated ?? false;

            // Generate IDs based on user state
            string userId = isAuthenticated ? User.FindFirstValue(ClaimTypes.NameIdentifier)! : "guest_" + Guid.NewGuid().ToString();

            // Utilize ASP.NET Session for Guest temporary continuity
            string sessionId = HttpContext.Session.GetString("TempSessionId") ?? Guid.NewGuid().ToString();
            if (!isAuthenticated && HttpContext.Session.GetString("TempSessionId") == null)
            {
                HttpContext.Session.SetString("TempSessionId", sessionId);
            }

            try
            {
                string aiFullResponse = "";

                // Stream response from Python Core
                await foreach (var chunk in _chatGrpcClient.StreamChatAsync(userId, sessionId, prompt, cancellationToken))
                {
                    aiFullResponse += chunk;
                    await Response.WriteAsync($"data: {chunk}\n\n");
                    await Response.Body.FlushAsync(cancellationToken);
                }

                // ONLY save to Database if the user is authenticated (Logged in)
                if (isAuthenticated)
                {
                    await _chatSessionService.SaveMessagesAsync(sessionId, prompt, aiFullResponse);
                }
            }
            catch (Exception ex)
            {
                await Response.WriteAsync($"data: [SYSTEM ERROR: {ex.Message}]\n\n");
            }
        }
    }
}