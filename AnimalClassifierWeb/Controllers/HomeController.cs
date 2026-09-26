using AnimalClassifierWeb.Services;
using Microsoft.AspNetCore.Mvc;

namespace AnimalClassifierWeb.Controllers
{
    public class HomeController : Controller
    {
        private readonly AnimalClassifierService _classifier;

        public HomeController(AnimalClassifierService classifier)
        {
            _classifier = classifier;
        }

        [HttpGet]
        public IActionResult Index()
        {
            return View();
        }

        [HttpPost]
        public async Task<IActionResult> Index(
            IFormFile image,
            string imageUrl)
        {
            try
            {
                byte[] imageBytes;

                if (image != null && image.Length > 0)
                {
                    using var memoryStream = new MemoryStream();

                    await image.CopyToAsync(memoryStream);

                    imageBytes = memoryStream.ToArray();
                }
                else if (!string.IsNullOrWhiteSpace(imageUrl))
                {
                    if (!Uri.TryCreate(imageUrl, UriKind.Absolute, out var uri))
                    {
                        ViewBag.Error = "Некоректне посилання на зображення.";
                        return View();
                    }

                    using var httpClient = new HttpClient();

                    httpClient.Timeout = TimeSpan.FromSeconds(20);

                    httpClient.DefaultRequestHeaders.UserAgent.ParseAdd(
                        "AnimalClassifierWeb/1.0"
                    );

                    var response = await httpClient.GetAsync(uri);

                    if (!response.IsSuccessStatusCode)
                    {
                        ViewBag.Error =
                            $"Не вдалося завантажити зображення. Сервер повернув: {(int)response.StatusCode} ({response.StatusCode}).";

                        return View();
                    }

                    var contentType = response.Content.Headers.ContentType?.MediaType;

                    if (contentType == null || !contentType.StartsWith("image/"))
                    {
                        ViewBag.Error =
                            "Посилання не веде безпосередньо на зображення.";

                        return View();
                    }

                    imageBytes = await response.Content.ReadAsByteArrayAsync();
                }
                else
                {
                    ViewBag.Error =
                        "Оберіть зображення або вставте посилання.";

                    return View();
                }

                if (imageBytes.Length < 1000)
                {
                    ViewBag.Error =
                        "Зображення занадто маленьке або пошкоджене.";

                    return View();
                }

                await Task.Run(() =>
                {
                    var result = _classifier.Predict(imageBytes);

                    ViewBag.PredictedLabel = result.Label;
                    ViewBag.Confidence = result.Confidence;
                });

                return View();
            }
            catch (HttpRequestException ex)
            {
                ViewBag.Error =
                    $"Не вдалося завантажити зображення: {ex.Message}";

                return View();
            }
            catch (Exception ex)
            {
                ViewBag.Error =
                    $"Помилка: {ex.Message}";

                return View();
            }
        }
    }
}