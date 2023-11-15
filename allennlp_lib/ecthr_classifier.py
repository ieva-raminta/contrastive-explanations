from typing import Dict, Optional

from overrides import overrides
import torch

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import (
    FeedForward,
    Seq2SeqEncoder,
    Seq2VecEncoder,
    TextFieldEmbedder,
)
from allennlp.nn import InitializerApplicator, util
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.metrics.fbeta_multi_label_measure import F1MultiLabelMeasure

"""
shared things:
- "encoded_representations" field in output if "output_hidden_states" is turned on
- "linear_classifier_weight" field with w
- "linear_classifier_bias" field with b
"""


@Model.register("ecthr")
class ECtHRClassifier(Model):
    """
    This `Model` implements a basic text classifier. After embedding the text into
    a text field, we will optionally encode the embeddings with a `Seq2SeqEncoder`. The
    resulting sequence is pooled using a `Seq2VecEncoder` and then passed tocontrastive_projection
    a linear classification layer, which projects into the label space. If a
    `Seq2SeqEncoder` is not provided, we will pass the embedded text directly to the
    `Seq2VecEncoder`.

    Registered as a `Model` with name "basic_classifier".

    # Parameters

    vocab : `Vocabulary`
    text_field_embedder : `TextFieldEmbedder`
        Used to embed the input text into a `TextField`
    seq2seq_encoder : `Seq2SeqEncoder`, optional (default=`None`)
        Optional Seq2Seq encoder layer for the input text.
    seq2vec_encoder : `Seq2VecEncoder`
        Required Seq2Vec encoder layer. If `seq2seq_encoder` is provided, this encoder
        will pool its output. Otherwise, this encoder will operate directly on the output
        of the `text_field_embedder`.
    feedforward : `FeedForward`, optional, (default = `None`)
        An optional feedforward layer to apply after the seq2vec_encoder.
    dropout : `float`, optional (default = `None`)
        Dropout percentage to use.
    num_labels : `int`, optional (default = `None`)
        Number of labels to project to in classification layer. By default, the classification layer will
        project to the size of the vocabulary namespace corresponding to labels.
    label_namespace : `str`, optional (default = `"labels"`)
        Vocabulary namespace corresponding to labels. By default, we use the "labels" namespace.
    initializer : `InitializerApplicator`, optional (default=`InitializerApplicator()`)
        If provided, will be used to initialize the model parameters.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        seq2vec_encoder: Seq2VecEncoder,
        output_hidden_states: bool = False,
        feedforward: Optional[FeedForward] = None,
        dropout: float = None,
        num_labels: int = 3,
        label_namespace: str = "labels",
        namespace: str = "tokens",
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)
        self._text_field_embedder = text_field_embedder

        self._seq2vec_encoder = seq2vec_encoder
        self._feedforward = feedforward
        if feedforward is not None:
            self._classifier_input_dim = self._feedforward.get_output_dim()
        else:
            self._classifier_input_dim = self._seq2vec_encoder.get_output_dim()

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None
        self._label_namespace = label_namespace
        self._namespace = namespace

        if num_labels:
            self._num_labels = num_labels
        else:
            self._num_labels = vocab.get_vocab_size(namespace=self._label_namespace)
        self._classification_layer = torch.nn.Linear(self._classifier_input_dim, 17* self._num_labels)

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        initializer(self)

        self._output_hidden_states = output_hidden_states

    def merge_masks(self, mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
        return mask1 * (mask2 + 1)

    def forward(  # type: ignore
        self,
        facts: TextFieldTensors,
        labels: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        # Parameters

        tokens : `TextFieldTensors`
            From a `TextField`
        label(n) : `torch.IntTensor`, optional (default = `None`)
            From a `LabelField`

        # Returns

        An output dictionary consisting of:

            - `logits` (`torch.FloatTensor`) :
                A tensor of shape `(batch_size, num_labels)` representing
                unnormalized log probabilities of the label.
            - `probs` (`torch.FloatTensor`) :
                A tensor of shape `(batch_size, num_labels)` representing
                probabilities of the label.
            - `loss` : (`torch.FloatTensor`, optional) :
                A scalar loss to be optimised.
        """

        embedded_text = self._text_field_embedder(facts, gradient_checkpointing=True)
        mask = get_text_field_mask(facts)

        global_attention_mask = torch.zeros(
            facts["tokens"]["mask"].shape, dtype=torch.long, device="cuda"
        )
        global_attention_mask[:, [0]] = 1

        attention_mask = self.merge_masks(mask, global_attention_mask)

        embedded_text = self._seq2vec_encoder(embedded_text, mask=attention_mask)

        if self._dropout:
            embedded_text = self._dropout(embedded_text)

        if self._feedforward is not None:
            embedded_text = self._feedforward(embedded_text)

        logits = self._classification_layer(embedded_text)

        # logits.reshape(tokens.shape[0], -1, 3)
        probs = torch.softmax(logits, dim=-1)

        output_dict = {"logits": logits, "probs": probs}
        if self._output_hidden_states:
            output_dict["encoded_representations"] = embedded_text
        output_dict["token_ids"] = util.get_token_ids_from_text_field_tensors(facts)
        if labels is not None:
            loss = self._loss(output_dict["logits"].reshape([-1,3]), labels.long().view(-1)) 
            output_dict["loss"] = loss
        self._accuracy(torch.cat([logits]).reshape([-1,3]).to("cuda"), torch.cat([labels]).long().view(-1).to("cuda"))
        return output_dict

    @overrides
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the probabilities, converts index to string label, and
        add `"label"` key to the dictionary with the result.
        """
        predictions = output_dict["probs"]
        if predictions.dim() == 2:
            predictions_list = [predictions[i] for i in range(predictions.shape[0])]
        else:
            predictions_list = [predictions]
        classes = []
        for prediction in predictions_list:
            label_idx = prediction.argmax(dim=-1).item()
            label_str = self.vocab.get_index_to_token_vocabulary(
                self._label_namespace
            ).get(label_idx, str(label_idx))
            classes.append(label_str)
        output_dict["label"] = classes
        tokens = []
        for instance_tokens in output_dict["token_ids"]:
            tokens.append(
                [
                    self.vocab.get_token_from_index(
                        token_id.item(), namespace=self._namespace
                    )
                    for token_id in instance_tokens
                ]
            )
        output_dict["tokens"] = tokens
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {"accuracy": self._accuracy.get_metric(reset)}
        return metrics

    default_predictor = "ecthr"
