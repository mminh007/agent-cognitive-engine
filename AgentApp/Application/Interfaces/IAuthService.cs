using System.Threading.Tasks;
using Domain.Entities;

namespace Application.Interfaces
{
    public interface IAuthService
    {
        Task<User?> AuthenticateAsync(string username, string password);
        Task<User> RegisterAsync(string username, string password);
        string GenerateJwtToken(User user);
    }
}