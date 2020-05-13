# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team and Jangwon Park
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Optional, Union

import torch
import numpy as np

from transformers import (
    BasicTokenizer,
    PreTrainedTokenizer,
    Pipeline,
    ModelCard,
    is_tf_available,
    is_torch_available
)

from transformers.pipelines import ArgumentHandler

if is_tf_available():
    import tensorflow as tf

if is_torch_available():
    import torch


logger = logging.getLogger(__name__)


def custom_encode_plus(sentence,
                       tokenizer,
                       return_tensors=None):
    # {'input_ids': [2, 10841, 10966, 10832, 10541, 21509, 27660, 18, 3], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0]}
    words = sentence.split()

    tokens = []
    tokens_mask = []

    for word in words:
        word_tokens = tokenizer.tokenize(word)
        if not word_tokens:
            word_tokens = [tokenizer.unk_token]  # For handling the bad-encoded word
        tokens.extend(word_tokens)
        tokens_mask.extend([1] + [0] * (len(word_tokens) - 1))

    ids = tokenizer.convert_tokens_to_ids(tokens)
    len_ids = len(ids)
    total_len = len_ids + tokenizer.num_special_tokens_to_add()
    if tokenizer.max_len and total_len > tokenizer.max_len:
        ids, _, _ = tokenizer.truncate_sequences(
            ids,
            pair_ids=None,
            num_tokens_to_remove=total_len - tokenizer.max_len,
            truncation_strategy="longest_first",
            stride=0,
        )

    sequence = tokenizer.build_inputs_with_special_tokens(ids)
    token_type_ids = tokenizer.create_token_type_ids_from_sequences(ids)
    # HARD-CODED: As I know, most of the transformers architecture will be `[CLS] + text + [SEP]``
    #             Only way to safely cover all the cases is to integrate `token mask builder` in internal library.
    tokens_mask = [1] + tokens_mask + [1]
    words = [tokenizer.cls_token] + words + [tokenizer.sep_token]

    encoded_inputs = {}
    encoded_inputs["input_ids"] = sequence
    encoded_inputs["token_type_ids"] = token_type_ids

    if return_tensors == "tf" and is_tf_available():
        encoded_inputs["input_ids"] = tf.constant([encoded_inputs["input_ids"]])

        if "token_type_ids" in encoded_inputs:
            encoded_inputs["token_type_ids"] = tf.constant([encoded_inputs["token_type_ids"]])

        if "attention_mask" in encoded_inputs:
            encoded_inputs["attention_mask"] = tf.constant([encoded_inputs["attention_mask"]])

    elif return_tensors == "pt" and is_torch_available():
        encoded_inputs["input_ids"] = torch.tensor([encoded_inputs["input_ids"]])

        if "token_type_ids" in encoded_inputs:
            encoded_inputs["token_type_ids"] = torch.tensor([encoded_inputs["token_type_ids"]])

        if "attention_mask" in encoded_inputs:
            encoded_inputs["attention_mask"] = torch.tensor([encoded_inputs["attention_mask"]])

    elif return_tensors is not None:
        logger.warning(
            "Unable to convert output to tensors format {}, PyTorch or TensorFlow is not available.".format(
                return_tensors
            )
        )

    return encoded_inputs, words, tokens_mask


class NerPipeline(Pipeline):
    """
    Named Entity Recognition pipeline using ModelForTokenClassification head. See the
    `named entity recognition usage <../usage.html#named-entity-recognition>`__ examples for more information.

    This token recognition pipeline can currently be loaded from the :func:`~transformers.pipeline` method using
    the following task identifier(s):

    - "ner", for predicting the classes of tokens in a sequence: person, organisation, location or miscellaneous.

    The models that this pipeline can use are models that have been fine-tuned on a token classification task.
    See the list of available community models fine-tuned on such a task on
    `huggingface.co/models <https://huggingface.co/models?search=&filter=token-classification>`__.

    Arguments:
        model (:obj:`~transformers.PreTrainedModel` or :obj:`~transformers.TFPreTrainedModel`):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            :class:`~transformers.PreTrainedModel` for PyTorch and :class:`~transformers.TFPreTrainedModel` for
            TensorFlow.
        tokenizer (:obj:`~transformers.PreTrainedTokenizer`):
            The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
            :class:`~transformers.PreTrainedTokenizer`.
        modelcard (:obj:`str` or :class:`~transformers.ModelCard`, `optional`, defaults to :obj:`None`):
            Model card attributed to the model for this pipeline.
        framework (:obj:`str`, `optional`, defaults to :obj:`None`):
            The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be
            installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified
            and both frameworks are installed, will default to PyTorch.
        args_parser (:class:`~transformers.pipelines.ArgumentHandler`, `optional`, defaults to :obj:`None`):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (:obj:`int`, `optional`, defaults to :obj:`-1`):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, >=0 will run the model
            on the associated CUDA device id.
    """

    default_input_names = "sequences"

    def __init__(
        self,
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        tokenizer: PreTrainedTokenizer,
        modelcard: Optional[ModelCard] = None,
        framework: Optional[str] = None,
        args_parser: ArgumentHandler = None,
        device: int = -1,
        binary_output: bool = False,
        ignore_labels=["O"],
        task: str = "",
        ignore_special_tokens: bool = True
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            modelcard=modelcard,
            framework=framework,
            args_parser=args_parser,
            device=device,
            binary_output=binary_output,
            task=task,
        )

        self._basic_tokenizer = BasicTokenizer(do_lower_case=False)
        self.ignore_labels = ignore_labels
        self.ignore_special_tokens = ignore_special_tokens

    def __call__(self, *texts, **kwargs):
        inputs = self._args_parser(*texts, **kwargs)
        answers = []
        for sentence in inputs:

            # Manage correct placement of the tensors
            with self.device_placement():

                # [FIX] Split token by word-level
                tokens, words, tokens_mask = custom_encode_plus(
                    sentence,
                    self.tokenizer,
                    return_tensors=self.framework
                )

                # Forward
                if self.framework == "tf":
                    entities = self.model(tokens)[0][0].numpy()
                    input_ids = tokens["input_ids"].numpy()[0]
                else:
                    with torch.no_grad():
                        tokens = self.ensure_tensor_on_device(**tokens)
                        entities = self.model(**tokens)[0][0].cpu().numpy()
                        input_ids = tokens["input_ids"].cpu().numpy()[0]

            score = np.exp(entities) / np.exp(entities).sum(-1, keepdims=True)
            labels_idx = score.argmax(axis=-1)

            token_level_answer = []
            for idx, label_idx in enumerate(labels_idx):
                # NOTE Append every answer even though the `entity` is in `ignore_labels`
                token_level_answer += [
                    {
                        "word": self.tokenizer.convert_ids_to_tokens(int(input_ids[idx])),
                        "score": score[idx][label_idx].item(),
                        "entity": self.model.config.id2label[label_idx],
                    }
                ]

            # [FIX] Now let's change it to word-level NER
            word_idx = 0
            word_level_answer = []

            # NOTE: Might not be safe. BERT, ELECTRA etc. won't make issues.
            if self.ignore_special_tokens:
                words = words[1:-1]
                tokens_mask = tokens_mask[1:-1]
                token_level_answer = token_level_answer[1:-1]

            for mask, ans in zip(tokens_mask, token_level_answer):
                if mask == 1:
                    ans["word"] = words[word_idx]
                    word_idx += 1
                    if ans["entity"] not in self.ignore_labels:
                        word_level_answer.append(ans)

            # Append
            answers += [word_level_answer]
        if len(answers) == 1:
            return answers[0]
        return answers
