{
    "data_loader": {
        "batch_sampler": {
            "batch_size": 1,
            "type": "bucket"
        }
    },
    "dataset_reader": {
        "token_indexers": {
            "tokens": {
                "max_length": 4096,
                "model_name": "allenai/longformer-base-4096",
                "type": "pretrained_transformer"
            }
        },
        "tokenizer": {
            "add_special_tokens": false,
            "model_name": "allenai/longformer-base-4096",
            "type": "pretrained_transformer"
        },
        "type": "ecthr"
    },
    "evaluate_on_test": true,
    "model": {
        "dropout": 0.2,
        "feedforward": {
            "activations": "tanh",
            "hidden_dims": 50,
            "input_dim": 768,
            "num_layers": 1
        },
        "namespace": "tags",
        "seq2vec_encoder": {
            "embedding_dim": 768,
            "type": "cls_pooler"
        },
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "max_length": 4096,
                    "model_name": "allenai/longformer-base-4096",
                    "type": "pretrained_transformer"
                }
            }
        },
        "type": "ecthr"
    },
    "test_data_path": "data/ecthr/outcome/simple_test.jsonl",
    "train_data_path": "data/ecthr/outcome/simple_train.jsonl",
    "trainer": {
        "cuda_device": 0,
        "learning_rate_scheduler": {
            "cut_frac": 0.06,
            "type": "slanted_triangular"
        },
        "num_epochs": 10,
        "optimizer": {
            "lr": 3e-06,
            "type": "huggingface_adamw",
            "weight_decay": 0.1
        },
        "patience": 5,
        "use_amp": true,
        "validation_metric": "+loss"
    },
    "validation_data_path": "data/ecthr/outcome/simple_val.jsonl"
}