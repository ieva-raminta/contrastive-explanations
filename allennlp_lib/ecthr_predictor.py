from typing import List, Dict

import numpy
from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from allennlp.data.fields import MultiLabelField, LabelField
from allennlp.common.util import JsonDict, sanitize


@Predictor.register("ecthr")
class ECtHRPredictor(Predictor):
    """
    Predictor for the [`DecomposableAttention`](../models/decomposable_attention.md) model.

    Registered as a `Predictor` with name "outcome".
    """

    def predict(self, facts: str) -> JsonDict:
        """
        Predicts multilabel outcome for a given text of facts.

        # Parameters

        facts : `str`
            A passage representing what is assumed to be true.

        # Returns

        `JsonDict`
            A dictionary where the key "label_probs" determines the probabilities of each of
            a number of possible labels.
        """
        return self.predict_json({"facts": facts})

    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        return self.predict_instance(instance)

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        return sanitize(outputs)

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like `{"facts": "...", }`.
        """
        facts = json_dict["facts"]
        outcomes = json_dict["outcomes"]
        claims = json_dict["claims"]
        return self._dataset_reader.text_to_instance(facts=facts, outcomes=outcomes, claims=claims)

    @overrides
    def predictions_to_labeled_instances(
        self, instance: Instance, outputs: Dict[str, numpy.ndarray]
    ) -> List[Instance]:
        new_instance = instance.duplicate()
        # get multiclass multilabel labels from outputs
        labels = [numpy.argmax(output) for output in outputs["probs"]]
        #label = numpy.argmax(outputs["probs"])
        # Skip indexing, we have integer representations of the strings "entailment", etc.
        new_instance.add_field("labels", [LabelField(label, skip_indexing=True) for label in labels])
        return [new_instance]
