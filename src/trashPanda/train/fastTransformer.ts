// TODO: Setup a diffusion training in tfjs

import * as tf from "@tensorflow/tfjs";
import axios from 'axios';
import { JSDOM } from 'jsdom';
import { FastTransformer } from './models/fastTransformer';

// Constants
const CHUNK_SIZE = 200;
const QUERY_TERM = 'consciousness';
const URL = `https://arxiv.org/search/?query=${QUERY_TERM}&searchtype=all`;

// Function to fetch data from arXiv
async function fetchData() {
  const response = await axios.get(URL);
  const dom = new JSDOM(response.data);
  const results = dom.window.document.querySelectorAll(".arxiv-result .mathjax");
  let textChunks = [];

  results.forEach(result => {
    const text = result.textContent;
    for (let i = 0; i < text.length; i += CHUNK_SIZE) {
      textChunks.push(text.slice(i, i + CHUNK_SIZE));
    }
  });

  return textChunks;
}

// Function to preprocess data
function preprocessData(data: string[]) {
  // TODO: Tokenization and data preprocessing to match the model's input format
  return data;
}

// Model configuration
const modelConfig = {
  modelDim: 64,
  attn_heads: 8,
  attn_head_dim: 64,
  dropout: 0,
  depth: 6,
};

// Main training function
async function trainModel() {
  const dataChunks = await fetchData();
  const processedData = preprocessData(dataChunks);

  const transformer = new Transformer(modelConfig);

  // Compile the model
  transformer.layers.compile({
    optimizer: tf.train.adam(),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  // TODO: Format the processedData into appropriate input and targets
  const inputs = tf.tensor(processedData);
  const targets = /* Targets corresponding to the inputs */;

  // Training the model
  transformer.layers.fit(inputs, targets, {
    epochs: 10,
    batchSize: 32,
  });
}

// Starting the training
trainModel();
