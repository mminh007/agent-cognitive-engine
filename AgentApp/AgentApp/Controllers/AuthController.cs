using System;
using System.Threading.Tasks;
using Application.DTOs;
using Application.Interfaces;
using Microsoft.AspNetCore.Mvc;

namespace AgentApp.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class AuthController : ControllerBase
    {
        private readonly IAuthService _authService;

        // Inject the Authentication Service via constructor
        public AuthController(IAuthService authService)
        {
            _authService = authService;
        }

        [HttpPost("register")]
        public async Task<IActionResult> Register([FromBody] RegisterRequest request)
        {
            if (!ModelState.IsValid)
            {
                return BadRequest(ModelState);
            }

            try
            {
                // Note: In a production environment, you should check if the username already exists 
                // inside the AuthService before attempting to create a new user.
                var newUser = await _authService.RegisterAsync(request.Username, request.Password);

                return Ok(new
                {
                    Message = "User registered successfully.",
                    UserId = newUser.Id
                });
            }
            catch (Exception ex)
            {
                // Log the exception here
                return StatusCode(500, new { Message = "An error occurred during registration.", Details = ex.Message });
            }
        }

        [HttpPost("login")]
        public async Task<IActionResult> Login([FromBody] LoginRequest request)
        {
            if (!ModelState.IsValid)
            {
                return BadRequest(ModelState);
            }

            // Authenticate the user by verifying credentials against the hashed password in the database
            var user = await _authService.AuthenticateAsync(request.Username, request.Password);

            if (user == null)
            {
                return Unauthorized(new { Message = "Invalid username or password." });
            }

            // Generate JWT Token for the authenticated user
            var token = _authService.GenerateJwtToken(user);

            // Return the token to the client. The frontend will store this (e.g., in localStorage or an HttpOnly cookie)
            // and attach it to the "Authorization: Bearer {token}" header for subsequent requests.
            return Ok(new
            {
                Message = "Login successful.",
                Token = token,
                UserId = user.Id,
                Username = user.Username
            });
        }
    }
}