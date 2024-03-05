local transformer_model = "allenai/longformer-base-4096";
local transformer_dim = 768;

{
  "dataset_reader": {
    "type": "ecthr",
    "tokenizer": {
      "type": "pretrained_transformer",
      "model_name": transformer_model,
      "add_special_tokens": false,
    },
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": transformer_model,
        "max_length": 4096,
      }
    }
  },
  "train_data_path": "data/ecthr/outcome/simple_train.jsonl",
  "validation_data_path": "data/ecthr/outcome/simple_val.jsonl",
  "test_data_path": "data/ecthr/outcome/simple_test.jsonl",
  evaluate_on_test: true,
  "model": {
    "type": "ecthr",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "pretrained_transformer",
          "model_name": transformer_model,
          "max_length": 4096,
        }
      }
    },
    "seq2vec_encoder": {
       "type": "cls_pooler",
       "embedding_dim": transformer_dim,
    },
    "feedforward": {
      "input_dim": transformer_dim,
      "num_layers": 1,
      "hidden_dims": 200,
      "activations": "tanh"
    },
    "dropout": 0.3,
    "namespace": "tags"
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size" : 1
    }
  },
  "trainer": {
    "num_epochs": 10,
    patience: 5,
    "cuda_device" : 0,
    "validation_metric": "+loss",
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "cut_frac": 0.06
    },
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 3e-5,
      "weight_decay": 0.1,
    },
    "use_amp": true,
  }
}
