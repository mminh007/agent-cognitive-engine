using Microsoft.AspNetCore.Mvc;

namespace AgentApp.Controllers
{
    public class ViewController : Controller
    {
        // Target URL: https://localhost:xxxx/ OR https://localhost:xxxx/Home
        [HttpGet]
        [Route("")]
        [Route("Home")]
        public IActionResult Index()
        {
            // By convention, this automatically renders: Views/View/Index.cshtml
            return View();
        }

        // Target URL: https://localhost:xxxx/Auth/Login
        [HttpGet]
        [Route("Auth/Login")]
        public IActionResult Login()
        {
            // By convention, this automatically renders: Views/View/Login.cshtml
            return View();
        }

        // Target URL: https://localhost:xxxx/Privacy
        [HttpGet]
        [Route("Privacy")]
        public IActionResult Privacy()
        {
            // By convention, this automatically renders: Views/View/Privacy.cshtml
            return View();
        }
    }
}