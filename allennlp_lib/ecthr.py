import itertools
from typing import Dict, Optional
import json
import logging

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import (
    Field,
    TextField,
    LabelField,
    MultiLabelField,
    MetadataField,
)
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import (
    Tokenizer,
    SpacyTokenizer,
    PretrainedTransformerTokenizer,
)

from typing import List

logger = logging.getLogger(__name__)


@DatasetReader.register("ecthr")
class ECtHRReader(DatasetReader):
    """
    Reads a file from the Stanford Natural Language Inference (ECtHR) dataset.  This data is
    formatted as jsonl, one json-formatted instance per line.  The keys in the data are
    "facts", "claims", and "outcomes",
    "silver_rationales", "gold_rationales".  We convert these keys into fields named "facts",
    "violated_artciles" and "allegedly_violated_artciles",
    along with a metadata field containing the tokenized strings of the
    facts.

    Registered as a `DatasetReader` with name "ECtHR".

    # Parameters

    tokenizer : `Tokenizer`, optional (default=`SpacyTokenizer()`)
        We use this `Tokenizer` for both the premise and the hypothesis.  See :class:`Tokenizer`.
    token_indexers : `Dict[str, TokenIndexer]`, optional (default=`{"tokens": SingleIdTokenIndexer()}`)
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
    combine_input_fields : `bool`, optional
            (default=`isinstance(tokenizer, PretrainedTransformerTokenizer)`)
        If False, represent the premise and the hypothesis as separate fields in the instance.
        If True, tokenize them together using `tokenizer.tokenize_sentence_pair()`
        and provide a single `tokens` field in the instance.
    """

    def __init__(
        self,
        tokenizer: Optional[Tokenizer] = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        combine_input_fields: Optional[bool] = None,
        **kwargs,
    ) -> None:
        super().__init__(manual_distributed_sharding=True, **kwargs)
        self._tokenizer = tokenizer or SpacyTokenizer()
        if isinstance(self._tokenizer, PretrainedTransformerTokenizer):
            assert not self._tokenizer._add_special_tokens
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        if combine_input_fields is not None:
            self._combine_input_fields = combine_input_fields
        else:
            self._combine_input_fields = isinstance(
                self._tokenizer, PretrainedTransformerTokenizer
            )

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        import torch.distributed as dist
        from allennlp.common.util import is_distributed

        if is_distributed():
            start_index = dist.get_rank()
            step_size = dist.get_world_size()
            logger.info(
                "Reading ECtHR instances %% %d from jsonl dataset at: %s",
                step_size,
                file_path,
            )
        else:
            start_index = 0
            step_size = 1
            logger.info("Reading ECtHR instances from jsonl dataset at: %s", file_path)

        with open(file_path, "r") as ECtHR_file:
            example_iter = (json.loads(line) for line in ECtHR_file)
            filtered_example_iter = (
                example for example in example_iter if example["claims"] != "-"
            )
            for example in itertools.islice(
                filtered_example_iter, start_index, None, step_size
            ):
                outcomes = example["outcomes"]
                claims = example["claims"]
                facts = example["facts"]
                yield self.text_to_instance(
                    facts, outcomes, claims
                )

    @overrides
    def text_to_instance(
        self,  # type: ignore
        facts: List[str],
        outcomes: List[str],
        claims: List[str],
    ) -> Instance:
        fields: Dict[str, Field] = {}
        facts = self._tokenizer.tokenize(" ".join(facts))

        facts_tokens = self._tokenizer.add_special_tokens(facts)
        fields["facts"] = TextField(facts_tokens, self._token_indexers)

        metadata = {
            "facts_tokens": [x.text for x in facts_tokens],
        }
        fields["metadata"] = MetadataField(metadata)

        labels = []
        for i, claim in enumerate(claims): 
            if claim == 0: 
                labels.append("not_claimed")
            elif claim == 1 and outcomes[i] == 0: 
                labels.append("claimed_not_violated")
            elif claim == 1 and outcomes[i] == 1: 
                labels.append("claimed_and_violated")

        fields["label"] = [LabelField(label) for label in labels]

        return Instance(fields)
