from typing import List, Dict
import pandas as pd
import torch
import math
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


class Perplexity:
    def __init__(self, model_name):
        """

        :param model_name: Huggingface model name
        """
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

    def _compute_token_probability(self, text):

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
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

        # Case when there's only one token prob
        if type(token_probs.squeeze().tolist()) == float:
            token_probs = list([token_probs.squeeze().tolist()])
        else:
            token_probs = token_probs.squeeze().tolist()

        return list(zip(tokens[1:], token_probs))

    def _process_data(self, data):
        token_to_prob = {}

        for index, d in enumerate(data):
            token_probs = self._compute_token_probability(d)
            token_to_prob[index] = token_probs

        return token_to_prob

    def _calculate_sentence_probability(self, token_probs):
        log_probs = [math.log(prob) for prob in token_probs]
        sentence_probability = sum(log_probs)
        return sentence_probability

    def _calculate_perplexity(self, sentence_log_prob, num_tokens):
        return math.exp(-sentence_log_prob / num_tokens)

    # def _batch_process(self, data):
    #     for i in range(0, len(data), self.batch_size):
    #         yield data[i: i + self.batch_size]

    def _calculate_perplexity_overall(self, data):

        overall_perplexity = []

        for text in tqdm(data):
            batch_data = [text]
            batch_results = self._process_data(batch_data)
            for idx, tokens in batch_results.items():
                token_probabilities = [prob for _, prob in tokens]
                sentence_prob = self._calculate_sentence_probability(token_probabilities)
                perplexity_value = self._calculate_perplexity(sentence_prob, len(token_probabilities))
                overall_perplexity.append(perplexity_value)

        return overall_perplexity

    def get_score(self, texts: List[str]) -> float:
        """
        Calculate the perplexity score for a given HF model and a list of texts.
        :param texts: List of strings.
        :return: Float, the perplexity score.
        """

        overall_perplexity_val = self._calculate_perplexity_overall(texts)
        avg_perplexity_val = sum(overall_perplexity_val) / len(overall_perplexity_val)
        return avg_perplexity_val


if __name__ == "__main__":
    model = "unsloth/Llama-3.2-3B-Instruct"
    # model = 'meta-llama/Llama-3.2-1B-Instruct'
    data = pd.read_csv('../../data/test.csv')
    def concat(x):
        return x['instruction'] + "\n" + x['output']
    texts = data.apply(lambda x: concat(x), axis=1)
    perplexity = Perplexity(model)
    score = perplexity.get_score(texts)
    print(f"Perplexity Score: {score}")
