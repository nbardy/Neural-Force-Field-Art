import * as tfvis from "@tensorflow/tfjs-vis";
import * as tf from "@tensorflow/tfjs";

// const surface = tfvis
//   .visor()
//   .surface({ name: "histos", tab: "traced models i/o" });

/**
 * Wraps a model to log histograms for each input and output of each layer's execution.
 * @param {tf.Model} originalModel - The original TensorFlow model.
 * @returns {tf.Model} - The wrapped model with histogram logging functionality.
 */
export function modelTraced(originalModel) {
  // Get the layers from the original model.
  const layers = originalModel.layers;

  // Wrap each layer to log histograms.
  layers.forEach((layer, idx) => {
    const originalCall = layer.call;

    layer.call = function (inputs, kwargs) {
      const outputs = originalCall.apply(layer, [inputs, kwargs]);
      const data = [
        { index: 0, value: 50 },
        { index: 1, value: 100 },
        { index: 2, value: 150 },
      ];

      // Log histogram for the input/output of each layer.
      // const surfaceIn = { name: `Layer ${idx + 1} Input`, tab: "Histograms" };
      // tfvis.render.histogram(surface, inputs.dataSync());
      // tfvis.render.histogram(surface, outputs.dataSync());

      // tfvis.render.barchart(surface, data, {});

      return outputs;
    };
  });

  return originalModel;
}

/**
 * Takes a list of layers, builds it sequentially, and adds histogram tracing functionality.
 * @param {Array} layers - List of layers for the model.
 * @returns {tf.Model} - The constructed model with histogram logging functionality.
 */
export function sequentialTraced(layers) {
  // Construct a model using the provided layers.
  const model = tf.sequential({ layers });

  // Add histogram tracing functionality to the model.
  // return modelTraced(model);
  return model;
}
