using System;
using System.Text;
using System.Threading.RateLimiting;
using Application.Interfaces;
using Application.Services;
using Infrastructure.Data;
using Infrastructure.Interfaces;
using Infrastructure.Services;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.RateLimiting;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.IdentityModel.Tokens;

namespace AgentApp
{
    public class Program
    {
        public static void Main(string[] args)
        {
            var builder = WebApplication.CreateBuilder(args);

            // 1. Database Configuration
            builder.Services.AddDbContext<AppDbContext>(options =>
                options.UseSqlServer(builder.Configuration.GetConnectionString("DefaultConnection")));

            // 2. Dependency Injection (Strict Interface Bindings)
            builder.Services.AddScoped<IAuthService, AuthService>();
            builder.Services.AddScoped<IChatSessionService, ChatSessionService>();
            builder.Services.AddScoped<IChatGrpcClient, ChatGrpcClient>();

            // 3. JWT Authentication Configuration
            var jwtSecret = builder.Configuration["JwtSettings:Secret"] ?? "YourSuperSecretKey_MustBeAtLeast32CharsLong!";
            builder.Services.AddAuthentication(options =>
            {
                options.DefaultAuthenticateScheme = JwtBearerDefaults.AuthenticationScheme;
                options.DefaultChallengeScheme = JwtBearerDefaults.AuthenticationScheme;
            })
            .AddJwtBearer(options =>
            {
                options.TokenValidationParameters = new TokenValidationParameters
                {
                    ValidateIssuerSigningKey = true,
                    IssuerSigningKey = new SymmetricSecurityKey(Encoding.ASCII.GetBytes(jwtSecret)),
                    ValidateIssuer = false,
                    ValidateAudience = false
                };
            });

            // 4. ASP.NET Session Configuration (For Guest continuity)
            builder.Services.AddDistributedMemoryCache();
            builder.Services.AddSession(options =>
            {
                options.IdleTimeout = TimeSpan.FromMinutes(30);
                options.Cookie.HttpOnly = true;
                options.Cookie.IsEssential = true;
            });

            // 5. Rate Limiting Configuration
            builder.Services.AddRateLimiter(options =>
            {
                options.AddFixedWindowLimiter("ChatStreamPolicy", opt =>
                {
                    opt.Window = TimeSpan.FromSeconds(10);
                    opt.PermitLimit = 5;
                    opt.QueueProcessingOrder = QueueProcessingOrder.OldestFirst;
                    opt.QueueLimit = 2;
                });

                options.OnRejected = async (context, token) =>
                {
                    context.HttpContext.Response.StatusCode = 429;
                    await context.HttpContext.Response.WriteAsync("System overloaded. Please slow down.", token);
                };
            });

            builder.Services.AddControllersWithViews();

            var app = builder.Build();
            app.UseStaticFiles();

            // Middleware Pipeline Setup
            app.UseRouting();

            // Enable Authentication & Session
            app.UseSession();
            app.UseAuthentication();
            app.UseAuthorization();

            app.UseRateLimiter();

            app.MapControllers().RequireRateLimiting("ChatStreamPolicy");
            app.MapControllerRoute(
                name: "default",
                pattern: "{controller=Home}/{action=Index}/{id?}");

            app.Run();
        }
    }
}