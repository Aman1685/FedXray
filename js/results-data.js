
/*
 * results-data.js
 * Official NIH ChestX-ray14 test-set results.
 *
 * Source:
 * results/evaluation/official_test/all_models_summary.csv
 *
 * Evaluation:
 * - 25,596 official test images
 * - Default threshold: 0.5
 * - Seven trained checkpoints
 *
 * Metric mapping:
 * accuracy  = Per_Label_Accuracy
 * precision = Macro_Precision
 * recall    = Macro_Recall
 * f1        = Macro_F1
 * auc       = Macro_ROC_AUC
 *
 * All values are actual evaluation results, not estimates.
 */

const EVALUATION_RESULTS = {
  description:
    "Official NIH ChestX-ray14 test-set evaluation using 25,596 images and a fixed threshold of 0.5 for every disease. Results are from the seven best checkpoints. Accuracy represents per-label accuracy; precision, recall and F1 are macro-averaged across the 14 disease labels. AUC is macro ROC-AUC. All metrics are represented as decimals from 0 to 1.",

  series: [
    {
      model: "centralized",
      label: "Centralized CNN",
      metrics: {
        accuracy: 0.6591989819837921,
        precision: 0.1164770451184417,
        recall: 0.544438118635218,
        f1: 0.1742105852113008,
        auc: 0.6849738503741977
      }
    },

    {
      model: "fedavg",
      label: "FedAvg - IID",
      metrics: {
        accuracy: 0.6734980912196102,
        precision: 0.0853571898525036,
        recall: 0.44369237133452394,
        f1: 0.1370531385810967,
        auc: 0.6322455837231393
      }
    },

    {
      model: "fedavg",
      label: "FedAvg - Non-IID",
      metrics: {
        accuracy: 0.6900743419730762,
        precision: 0.08678274157222339,
        recall: 0.4172160442407721,
        f1: 0.1355035363166475,
        auc: 0.6295821235145976
      }
    },

    {
      model: "fedavg",
      label: "FedAvg - Balanced Non-IID",
      metrics: {
        accuracy: 0.6999391646016119,
        precision: 0.08804610384298826,
        recall: 0.41182476796925754,
        f1: 0.13637377465736944,
        auc: 0.6360942726388231
      }
    },

    {
      model: "fedprox",
      label: "FedProx - IID",
      metrics: {
        accuracy: 0.6833936106088004,
        precision: 0.08579687174046588,
        recall: 0.4181205716726138,
        f1: 0.12769815967817996,
        auc: 0.6243517060177866
      }
    },

    {
      model: "fedprox",
      label: "FedProx - Non-IID",
      metrics: {
        accuracy: 0.7119918290804367,
        precision: 0.08705313925146121,
        recall: 0.37220283371237467,
        f1: 0.12348610310784905,
        auc: 0.629926350692827
      }
    },

    {
      model: "fedprox",
      label: "FedProx - Balanced Non-IID",
      metrics: {
        accuracy: 0.7044515884178332,
        precision: 0.08776230554040618,
        recall: 0.39467578031201195,
        f1: 0.12954117165131138,
        auc: 0.6301875884754126
      }
    }
  ]
};
