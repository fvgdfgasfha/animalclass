using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace AnimalClassifierWeb.MLModel
{
    public partial class MLModel1
    {
        public class ModelInput
        {
            [LoadColumn(0)]
            [ColumnName("Label")]
            public string Label { get; set; }

            [LoadColumn(1)]
            [ColumnName("ImageSource")]
            public byte[] ImageSource { get; set; }
        }

        public class ModelOutput
        {
            [ColumnName("Label")]
            public uint Label { get; set; }

            [ColumnName("ImageSource")]
            public byte[] ImageSource { get; set; }

            [ColumnName("PredictedLabel")]
            public string PredictedLabel { get; set; }

            [ColumnName("Score")]
            public float[] Score { get; set; }
        }

        private static readonly string MLNetModelPath =
            Path.Combine(
                AppContext.BaseDirectory,
                "MLModel",
                "MLModel1.mlnet"
            );

        private static readonly Lazy<PredictionEngine<ModelInput, ModelOutput>> PredictEngine =
            new Lazy<PredictionEngine<ModelInput, ModelOutput>>(
                CreatePredictEngine,
                true
            );

        private static PredictionEngine<ModelInput, ModelOutput> CreatePredictEngine()
        {
            var mlContext = new MLContext();

            ITransformer mlModel =
                mlContext.Model.Load(MLNetModelPath, out _);

            return mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(
                mlModel
            );
        }

        public static ModelOutput Predict(ModelInput input)
        {
            return PredictEngine.Value.Predict(input);
        }

        public static IOrderedEnumerable<KeyValuePair<string, float>> PredictAllLabels(
            ModelInput input)
        {
            var result = Predict(input);

            return GetSortedScoresWithLabels(result);
        }

        public static IOrderedEnumerable<KeyValuePair<string, float>> GetSortedScoresWithLabels(
            ModelOutput result)
        {
            if (result.Score == null || result.Score.Length == 0)
            {
                throw new Exception("Модель не повернула оцінки.");
            }

            var labels = GetLabels();

            var resultDictionary = new Dictionary<string, float>();

            for (int i = 0; i < labels.Count && i < result.Score.Length; i++)
            {
                resultDictionary[labels[i]] = result.Score[i];
            }

            return resultDictionary
                .OrderByDescending(x => x.Value);
        }

        private static List<string> GetLabels()
        {
            var schema = PredictEngine.Value.OutputSchema;

            var labelColumn = schema.GetColumnOrNull("Label");

            if (labelColumn == null)
            {
                throw new Exception(
                    "Label column not found in model."
                );
            }

            var keyNames = new VBuffer<ReadOnlyMemory<char>>();

            labelColumn.Value.GetKeyValues(ref keyNames);

            return keyNames
                .DenseValues()
                .Select(x => x.ToString())
                .ToList();
        }
    }
}