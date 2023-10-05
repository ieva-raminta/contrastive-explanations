from typing import List, Dict

import numpy
from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from allennlp.data.fields import MultiLabelField


@Predictor.register("outcome_fixed")
class OutcomePredictorFixed(Predictor):
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

    # def predict_json(self, js: JsonDict) -> JsonDict:
    #     return self.predict_json(js)

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like `{"facts": "...", }`.
        """
        facts_text = json_dict["facts"]
        return self._dataset_reader.text_to_instance(facts_text)

    @overrides
    def predictions_to_labeled_instances(
        self, instance: Instance, outputs: Dict[str, numpy.ndarray]
    ) -> List[Instance]:
        new_instance = instance.duplicate()
        # get multiclass multilabel labels from outputs
        labels = [numpy.argmax(output) for output in outputs["probs"]]
        #label = numpy.argmax(outputs["probs"])
        # Skip indexing, we have integer representations of the strings "entailment", etc.
        new_instance.add_field("labels", MultiLabelField(labels, skip_indexing=True))
        return [new_instance]
