from typing import Dict, Optional

from overrides import overrides
import torch

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
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
        seq2seq_encoder: Seq2SeqEncoder = None,
        feedforward: Optional[FeedForward] = None,
        dropout: float = None,
        num_labels: int = None,
        label_namespace: str = "label1",
        threshold: float = 0.5,
        namespace: str = "tokens",
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ) -> None:

        super().__init__(vocab, **kwargs)
        self._text_field_embedder = text_field_embedder

        if seq2seq_encoder:
            self._seq2seq_encoder = seq2seq_encoder
        else:
            self._seq2seq_encoder = None

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
        self._classification_layer1 = torch.nn.Linear(self._classifier_input_dim, self._num_labels)
        self._classification_layer2 = torch.nn.Linear(self._classifier_input_dim, self._num_labels)
        self._classification_layer3 = torch.nn.Linear(self._classifier_input_dim, self._num_labels)
        self._classification_layer4 = torch.nn.Linear(self._classifier_input_dim, self._num_labels)
        self._classification_layer5 = torch.nn.Linear(self._classifier_input_dim, self._num_labels)
        self._classification_layer6 = torch.nn.Linear(self._classifier_input_dim, self._num_labels)
        self._classification_layer7 = torch.nn.Linear(self._classifier_input_dim, self._num_labels)
        self._classification_layer8 = torch.nn.Linear(self._classifier_input_dim, self._num_labels)
        self._classification_layer9 = torch.nn.Linear(self._classifier_input_dim, self._num_labels)
        self._classification_layer10 = torch.nn.Linear(self._classifier_input_dim, self._num_labels)
        self._classification_layer11 = torch.nn.Linear(self._classifier_input_dim, self._num_labels)
        self._classification_layer12 = torch.nn.Linear(self._classifier_input_dim, self._num_labels)
        self._classification_layer13 = torch.nn.Linear(self._classifier_input_dim, self._num_labels)
        self._classification_layer14 = torch.nn.Linear(self._classifier_input_dim, self._num_labels)
        self._classification_layer15 = torch.nn.Linear(self._classifier_input_dim, self._num_labels)
        self._classification_layer16 = torch.nn.Linear(self._classifier_input_dim, self._num_labels)
        self._classification_layer17 = torch.nn.Linear(self._classifier_input_dim, self._num_labels)

        self._threshold = threshold
        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        initializer(self)

        self._output_hidden_states = output_hidden_states

    def forward(  # type: ignore
        self, facts: TextFieldTensors, label1: torch.IntTensor = None, label2: torch.IntTensor = None, label3: torch.IntTensor = None, label4: torch.IntTensor = None, label5: torch.IntTensor = None, label6: torch.IntTensor = None, label7: torch.IntTensor = None, label8: torch.IntTensor = None, label9: torch.IntTensor = None, label10: torch.IntTensor = None, label11: torch.IntTensor = None, label12: torch.IntTensor = None, label13: torch.IntTensor = None, label14: torch.IntTensor = None, label15: torch.IntTensor = None, label16: torch.IntTensor = None, label17: torch.IntTensor = None,
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
        embedded_text = self._text_field_embedder(facts)
        mask = get_text_field_mask(facts)

        if self._seq2seq_encoder:
            embedded_text = self._seq2seq_encoder(embedded_text, mask=mask)

        embedded_text = self._seq2vec_encoder(embedded_text, mask=mask)

        if self._dropout:
            embedded_text = self._dropout(embedded_text)

        if self._feedforward is not None:
            embedded_text = self._feedforward(embedded_text)

        logits1 = self._classification_layer1(embedded_text)
        logits2 = self._classification_layer2(embedded_text)
        logits3 = self._classification_layer3(embedded_text)
        logits4 = self._classification_layer4(embedded_text)
        logits5 = self._classification_layer5(embedded_text)
        logits6 = self._classification_layer6(embedded_text)
        logits7 = self._classification_layer7(embedded_text)
        logits8 = self._classification_layer8(embedded_text)
        logits9 = self._classification_layer9(embedded_text)
        logits10 = self._classification_layer10(embedded_text)
        logits11 = self._classification_layer11(embedded_text)
        logits12 = self._classification_layer12(embedded_text)
        logits13 = self._classification_layer13(embedded_text)
        logits14 = self._classification_layer14(embedded_text)
        logits15 = self._classification_layer15(embedded_text)
        logits16 = self._classification_layer16(embedded_text)
        logits17 = self._classification_layer17(embedded_text)

        # logits.reshape(tokens.shape[0], -1, 3)
        probs1 = torch.softmax(logits1, dim=-1) 
        probs2 = torch.softmax(logits2, dim=-1) 
        probs3 = torch.softmax(logits3, dim=-1) 
        probs4 = torch.softmax(logits4, dim=-1) 
        probs5 = torch.softmax(logits5, dim=-1) 
        probs6 = torch.softmax(logits6, dim=-1) 
        probs7 = torch.softmax(logits7, dim=-1) 
        probs8 = torch.softmax(logits8, dim=-1) 
        probs9 = torch.softmax(logits9, dim=-1) 
        probs10 = torch.softmax(logits10, dim=-1) 
        probs11 = torch.softmax(logits11, dim=-1) 
        probs12 = torch.softmax(logits12, dim=-1) 
        probs13 = torch.softmax(logits13, dim=-1) 
        probs14 = torch.softmax(logits14, dim=-1) 
        probs15 = torch.softmax(logits15, dim=-1) 
        probs16 = torch.softmax(logits16, dim=-1) 
        probs17 = torch.softmax(logits17, dim=-1) 

        output_dict1 = {"logits": logits1, "probs": probs1}
        output_dict2 = {"logits": logits2, "probs": probs2}
        output_dict3 = {"logits": logits3, "probs": probs3}
        output_dict4 = {"logits": logits4, "probs": probs4}
        output_dict5 = {"logits": logits5, "probs": probs5}
        output_dict6 = {"logits": logits6, "probs": probs6}
        output_dict7 = {"logits": logits7, "probs": probs7}
        output_dict8 = {"logits": logits8, "probs": probs8}
        output_dict9 = {"logits": logits9, "probs": probs9}
        output_dict10 = {"logits": logits10, "probs": probs10}
        output_dict11 = {"logits": logits11, "probs": probs11}
        output_dict12 = {"logits": logits12, "probs": probs12}
        output_dict13 = {"logits": logits13, "probs": probs13}
        output_dict14 = {"logits": logits14, "probs": probs14}
        output_dict15 = {"logits": logits15, "probs": probs15}
        output_dict16 = {"logits": logits16, "probs": probs16}
        output_dict17 = {"logits": logits17, "probs": probs17}

        label_dict = {"1": label1, "2": label2, "3": label3, "4": label4, "5": label5, "6": label6, "7": label7, "8": label8, "9": label9, "10": label10, "11": label11, "12": label12, "13": label13, "14": label14, "15": label15, "16": label16, "17": label17}
        output_dict = {}
        loss = 0

        for i,out_dict in enumerate([output_dict1, output_dict2, output_dict3, output_dict4, output_dict5, output_dict6, output_dict7, output_dict8, output_dict9, output_dict10, output_dict11, output_dict12, output_dict13, output_dict14, output_dict15, output_dict16, output_dict17]):
            if self._output_hidden_states:
                output_dict["encoded_representations"] = embedded_text
            output_dict["token_ids"] = util.get_token_ids_from_text_field_tensors(facts)
            if label_dict[i] is not None:
                loss += self._loss(out_dict["logits"], label_dict[i].long().view(-1))
                output_dict["accuracy"] += self._accuracy(out_dict["logits"], label_dict[i])
        output_dict["loss"] = loss
        output_dict["accuracy"] /= len(label_dict)
        self._accuracy = output_dict["accuracy"]

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
            label_str = self.vocab.get_index_to_token_vocabulary(self._label_namespace).get(
                label_idx, str(label_idx)
            )
            classes.append(label_str)
        output_dict["label"] = classes
        tokens = []
        for instance_tokens in output_dict["token_ids"]:
            tokens.append(
                [
                    self.vocab.get_token_from_index(token_id.item(), namespace=self._namespace)
                    for token_id in instance_tokens
                ]
            )
        output_dict["tokens"] = tokens
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {"accuracy": self._accuracy.get_metric(reset)}
        return metrics

    default_predictor = "ecthr"