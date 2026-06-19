using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using System.Linq;
using System.Security.Claims;
using System.Threading.Tasks;
using Infrastructure.Data;

namespace AgentApp.Controllers
{
    [ApiController]
    [Route("api/sessions")]
    [Authorize]
    public class SessionController : ControllerBase
    {
        private readonly AppDbContext _context;

        public SessionController(AppDbContext context)
        {
            _context = context;
        }

        [HttpGet("list")]
        public async Task<IActionResult> GetSessionList()
        {
            var userId = User.FindFirstValue(ClaimTypes.NameIdentifier);

            if (string.IsNullOrEmpty(userId))
            {
                return Unauthorized(new { Message = "User token is invalid." });
            }

            var sessions = await _context.ChatSessions
                .Where(s => s.UserId == userId)
                .OrderByDescending(s => s.LastUpdated) 
                .Select(s => new
                {
                    id = s.Id,
                    title = s.Title
                })
                .ToListAsync();

            return Ok(sessions);
        }
    }
}