from typing import Callable
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from .utils import exist_and_not_none, zero_pad_sequences
import numpy as np

def preprocess_data(data, input_template=None, prompt_key=None, chosen_key=None, rejected_key=None) -> str:
    # custom dataset
    if chosen_key and rejected_key:
        if prompt_key:
            prompt = data[prompt_key]
        else:
            prompt = ""
            input_template = None  # do not modified with input template again
        chosen = data[chosen_key]
        reject = data[rejected_key]
    else:
        # Anthropic/hh-rlhf
        # tasksource/oasst1_pairwise_rlhf_reward
        if exist_and_not_none(data, "chosen") and exist_and_not_none(data, "rejected"):
            prompt = data["prompt"] if exist_and_not_none(data, "prompt") else ""
            if prompt.startswith("prompter:"):
                prompt = (
                    prompt.replace("prompter:", "\nHuman: ").replace("assistant:", "\nAssistant: ") + "\nAssistant: "
                )
            chosen = data["chosen"]
            reject = data["rejected"]
            input_template = None  # do not modified with input template again
        # lvwerra/stack-exchange-paired
        elif exist_and_not_none(data, "response_j"):
            prompt = data["question"]
            chosen = data["response_j"]
            reject = data["response_k"]
        # lmsys/chatbot_arena_conversations
        elif exist_and_not_none(data, "winner") and exist_and_not_none(data, "conversation_a"):

            def process_chatbot_arena_conversations(lll):
                result = []
                for l in lll:
                    if "user" in l["role"]:
                        result.append(input_template.format(l["content"]))
                    else:
                        result.append(l["content"])
                return "\n".join(result)

            prompt = ""
            chosen = data["conversation_a"] if data["winner"] == "model_a" else data["conversation_b"]
            reject = data["conversation_b"] if data["winner"] == "model_a" else data["conversation_a"]
            chosen = process_chatbot_arena_conversations(chosen)
            reject = process_chatbot_arena_conversations(reject)
            input_template = None  # do not modified with input template again
        # openai/webgpt_comparisons
        elif exist_and_not_none(data, "answer_0") and exist_and_not_none(data, "answer_1"):
            prompt = data["question"]["full_text"]
            chosen = data["answer_0"] if data["score_0"] > data["score_1"] else data["answer_1"]
            reject = data["answer_1"] if data["score_0"] > data["score_1"] else data["answer_0"]
        # damo/CValues-Comparison https://www.modelscope.cn/datasets/damo/CValues-Comparison/quickstart
        elif exist_and_not_none(data, "pos_resp") and exist_and_not_none(data, "neg_resp"):
            prompt = data["prompt"]
            chosen = data["pos_resp"]
            reject = data["neg_resp"]
        else:
            raise ValueError("Unknown reward dataset")

    # margin loss
    margin = data["margin"] if exist_and_not_none(data, "margin") else 0

    # input template
    if input_template:
        prompt = input_template.format(prompt)
    return prompt, chosen, reject, margin


class RewardDataset(Dataset):
    """
    Dataset for reward model

    Args:
        dataset: dataset for reward model
        self.tokenizer: self.tokenizer for reward model
        self.max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        input_template="Human: {}\nAssistant: ",
        is_dpo=False,
    ) -> None:
        super().__init__()
        self.is_dpo = is_dpo

        self.prompts = []
        self.answers = []
        self.rewards = []
        if self.is_dpo:
            self.prompt_ids_lens = []
        else:
            self.margins = []

        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length
        self.is_dpo = is_dpo

        import json
        with open('/media/xuzhao/Partition5/qyzheng/qihang/Data/llmcaption/score.json', 'rb') as f:
            dataset = json.load(f)


        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
            prompt, answer, reward = data["prompt"], data["answer"], data["reward"]
            
            if np.isnan(reward):
                continue

            # prompt_ids_len for prompt mask
            if self.is_dpo:
                prompt_token = self.tokenizer(
                    prompt,
                    max_length=self.max_length,
                    padding=False,
                    truncation=True,
                    return_tensors="pt",
                )
                prompt_ids_len = prompt_token["attention_mask"].int().sum().item()
                # filter the sample whose length is greater than max_length (2 for answer length)
                if prompt_ids_len >= self.max_length - 2:
                    continue
                else:
                    self.prompt_ids_lens.append(prompt_ids_len)

            self.prompts.append(prompt)
            self.answers.append(answer[0])
            self.rewards.append(reward)
        
        #print(self.rewards)
        self.mean = np.mean(self.rewards)
        self.std = np.std(self.rewards)
        ##print(self.mean, self.std)
        #exit()

    def __len__(self):
        length = len(self.answers)
        return length

    def __getitem__(self, idx):
        prompt, answer, reward = self.prompts[idx], self.answers[idx], self.rewards[idx]
        if self.is_dpo:
            extra = self.prompt_ids_lens[idx]
        #else:
        #    extra = self.margins[idx]

        full_prompt = prompt + answer + " " + self.tokenizer.eos_token
        full_token = self.tokenizer(
            full_prompt,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )

        # to avoid EOS_token truncation
        full_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        full_token["attention_mask"][0][-1] = True

        return (
            full_token["input_ids"],
            full_token["attention_mask"],
            reward,
        )

    def collate_fn(self, item_list):
        input_ids = []
        input_masks = []
        rewards = []
        for input_id, input_mask, reward in item_list:
            reward = (reward - self.mean) / self.std
            input_ids.append(input_id)
            input_masks.append(input_mask)
            rewards.append(reward)

        input_ids = zero_pad_sequences(input_ids, value=self.tokenizer.pad_token_id)
        input_masks = zero_pad_sequences(input_masks)
        rewards = torch.tensor(rewards)
        return input_ids, input_masks, rewards
