from typing import List
import pandas as pd
import torch
import numpy as np
import math
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM


class Perplexity:
    def __init__(self, model_name):
        self.batch_size = 32

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS (Apple Silicon GPU)")

        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("Using CUDA:0")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.model.eval()
        self.model.to(self.device)

    def _compute_token_probability(self, sentence):
        inputs = self.tokenizer(sentence, return_tensors="pt").to(self.model.device)

        # Get model logits (output probabilities before softmax)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits

        # Shift logits and input_ids to align predicted tokens with actual tokens
        shifted_logits = logits[:, :-1, :]
        shifted_labels = inputs["input_ids"][:, 1:]

        # Compute probabilities for each token
        probs = torch.softmax(shifted_logits, dim=-1)

        # Get probabilities for the actual tokens in the sentence
        token_probs = torch.gather(probs, 2, shifted_labels.unsqueeze(-1)).squeeze(-1)

        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())

        # Exclude the first token <s> or [CLS]
        return list(zip(tokens[1:], token_probs.squeeze().tolist()))

    def _process_data(self, val_data):
        # for all rows
        token_to_prob = {}

        for index, row in val_data.iterrows():
            input_text = row["Instruction"]
            token_probs = self._compute_token_probability(input_text)
            token_to_prob[index] = token_probs

        return token_to_prob

    def _calculate_sentence_probability(self, token_probs):
        log_probs = [math.log(prob) for prob in token_probs]
        sentence_probability = sum(log_probs)
        return sentence_probability

    def _calculate_perplexity(self, sentence_log_prob, num_tokens):
        return math.exp(-sentence_log_prob / num_tokens)

    # Function to save checkpoint after each batch
    def _save_checkpoint(self, perplexity_values, batch_idx, checkpoint_file):
        checkpoint_data = {
            "perplexity_values": perplexity_values,
            "last_processed_batch": batch_idx,
        }
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f)

    # Function to load from the checkpoint
    def _load_checkpoint(self, checkpoint_file):
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, "r") as f:
                checkpoint_data = json.load(f)
                return checkpoint_data["perplexity_values"], checkpoint_data[
                    "last_processed_batch"
                ]
        else:
            return [], -1  # No checkpoint found, start from scratch

    def _batch_process(self, data):
        for i in range(0, len(data), self.batch_size):
            yield data[i : i + self.batch_size]

    def _process_data_in_batches(self, data):
        results = {}
        for batch_idx, batch_data in enumerate(self._batch_process(data)):
            batch_results = self._process_data(batch_data)
            results.update(batch_results)

        return results

    def _calculate_perplexity_overall(self, data, checkpoint_file=None):
        # Load the checkpoint if it exists
        # overall_perplexity, last_processed_batch = self._load_checkpoint(checkpoint_file)
        overall_perplexity = []

        # Start from the next batch after the checkpoint
        for batch_idx, batch_data in enumerate(self._batch_process(data)):

            batch_results = self._process_data(batch_data)
            for idx, tokens in batch_results.items():
                token_probabilities = [prob for _, prob in tokens]
                sentence_prob = self._calculate_sentence_probability(token_probabilities)
                perplexity_value = self._calculate_perplexity(sentence_prob, len(token_probabilities))
                overall_perplexity.append(perplexity_value)

            # # Save checkpoint after each batch
            # self._save_checkpoint(overall_perplexity, batch_idx, checkpoint_file)
            # print(f"Batch {batch_idx + 1} processed, perplexity checkpoint saved.")

        return overall_perplexity

    def get_score(self, texts: List[str]) -> float:
        """Calculates the perplexity score for a given HF model and a given dataset.

        Args:
            model (HF model): Model to calculate the perplexity score on.
            data (List[str]): List of strings on which the perplexity is calculated and averaged over.
        Returns:
            float: The perplexity score
        """

        overall_perplexity_val = self._calculate_perplexity_overall(texts)
        avg_perplexity_val = sum(overall_perplexity_val) / len(overall_perplexity_val)
        return avg_perplexity_val


if __name__ == "__main__":
    model = "meta-llama/Llama-3.2-1B-Instruct"
    data = pd.read_csv('../../data/test.csv')
    texts = data['output'].tolist()
    perplexity = Perplexity(model)
    perplexity.get_score(texts)