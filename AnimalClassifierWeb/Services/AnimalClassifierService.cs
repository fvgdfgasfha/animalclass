using AnimalClassifierWeb.MLModel;

namespace AnimalClassifierWeb.Services
{
    public class AnimalClassifierService
    {
        public AnimalPrediction Predict(byte[] imageBytes)
        {
            if (imageBytes == null || imageBytes.Length == 0)
            {
                throw new ArgumentException("Зображення порожнє.");
            }

            var input = new MLModel1.ModelInput
            {
                ImageSource = imageBytes
            };

            var result = MLModel1.Predict(input);

            var labels = MLModel1.GetSortedScoresWithLabels(result).ToList();

            if (labels.Count == 0)
            {
                throw new Exception("Модель не повернула результат.");
            }

            var bestResult = labels.First();

            return new AnimalPrediction
            {
                Label = bestResult.Key,
                Confidence = bestResult.Value * 100
            };
        }
    }

    public class AnimalPrediction
    {
        public string Label { get; set; }

        public float Confidence { get; set; }
    }
}