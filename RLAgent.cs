using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using StardewModdingAPI;

namespace fishing
{
    class RLAgent
    {
        private InferenceSession session;
        const string modelPath = "./Mods/RL-fishing/assets/policy_net.onnx";
        private IMonitor logger;

        public RLAgent(IMonitor monitor)
        {
            logger = monitor;

            SessionOptions options = new SessionOptions();
            options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR;
            options.AppendExecutionProvider_CPU(1);

            // Run inference
            session = new InferenceSession(modelPath, options);
        }

        public void reloadModel()
        {
            this.logger.Log("Model RELOADED", StardewModdingAPI.LogLevel.Trace);
            SessionOptions options = new SessionOptions();
            options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR;
            options.AppendExecutionProvider_CPU(1);
            session = new InferenceSession(modelPath, options);
        }

        public int Update(double[] currentState)
        {
            // 将输入数据从 double 转换为 float
            float[] floatState = Array.ConvertAll(currentState, x => (float)x);
            Tensor<float> input = new DenseTensor<float>(new[] { 1, 3 });

            input[0, 0] = floatState[0];
            input[0, 1] = floatState[1];
            input[0, 2] = floatState[2];

            // Setup inputs and outputs
            var inputs = new List<NamedOnnxValue>()
            {
                NamedOnnxValue.CreateFromTensor<float>("input", input) // 修改输入名称为 'input'
            };

            using (var results = session.Run(inputs))
            {
                Tensor<float> outputs = results.First().AsTensor<float>();

                var maxValue = outputs.Max();
                var maxIndex = outputs.ToList().IndexOf(maxValue);

                return maxIndex;
            }
        }
    }
}
