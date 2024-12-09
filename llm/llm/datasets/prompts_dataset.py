from torch.utils.data import Dataset
from tqdm import tqdm
from .utils import exist_and_not_none
import numpy as np

def preprocess_data(data, input_template=None, input_key=None, output_key=None):
    # custom dataset
    print(data.keys())
    if input_key and output_key:
        prompt = data[input_key]
        target = data[output_key]
    else:
        if type(data) == str:
            prompt = data
            target = None
            input_template = None        
        elif exist_and_not_none(data, "prompt") and exist_and_not_none(data, "chosen"):
            prompt = data["prompt"]
            target = data["chosen"]
            input_template = None  # do not modified with input template again
        # pvduy/sharegpt_alpaca_oa_vicuna_format
        elif exist_and_not_none(data, "prompt") and exist_and_not_none(data, "label"):
            prompt = data["prompt"].replace("USER:", "").replace("ASSISTANT:", "")
            target = data["label"].replace("</s>", "")
        # BelleGroup/train_0.5M_CN
        # LLMs/Alpaca-ShareGPT
        # yahma/alpaca-cleaned
        # QingyiSi/Alpaca-CoT
        elif exist_and_not_none(data, "instruction") and exist_and_not_none(data, "output"):
            prompt = (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{data['instruction']}\n\n### Input:\n{data['input']}\n\n### Response:\n"
            )
            #input = " " + data["input"] if exist_and_not_none(data, "input") else ""
            #prompt = data["instruction"] + input
            target = data["output"]
        # Open-Orca/OpenOrca
        elif exist_and_not_none(data, "system_prompt") and exist_and_not_none(data, "response"):
            prompt = data["system_prompt"] + "\n" + data["question"]
            target = data["response"]
        # crumb/gpt4all-clean
        # nomic-ai/gpt4all-j-prompt-generations
        elif exist_and_not_none(data, "prompt") and exist_and_not_none(data, "response"):
            prompt = data["prompt"]
            target = data["response"]
        # EleutherAI/pile [pretrain !!!]
        elif exist_and_not_none(data, "text") and exist_and_not_none(data, "meta"):
            assert input_template is None  # pretrain_mode
            prompt = ""
            target = data["text"]
        # for batch_inference.py
        elif exist_and_not_none(data, "input") and exist_and_not_none(data, "output"):
            prompt = data["input"]
            target = data["output"]
            input_template = None
        else:
            raise ValueError("Unknown SFT dataset")

    # input template
    if input_template:
        prompt = input_template.format(prompt)
    return prompt, target



def preprocess_data_etft(data, input_template=None, input_key=None, output_key=None):
    # custom dataset
    if input_key and output_key:
        prompt = data[input_key]
        target = data[output_key]
    else:
        if type(data) == str:
            prompt = data
            target = None
            input_template = None        
        elif exist_and_not_none(data, "prompt") and exist_and_not_none(data, "chosen"):
            prompt = data["prompt"]
            target = data["chosen"]
            input_template = None  # do not modified with input template again
        # pvduy/sharegpt_alpaca_oa_vicuna_format
        elif exist_and_not_none(data, "prompt") and exist_and_not_none(data, "label"):
            prompt = data["prompt"].replace("USER:", "").replace("ASSISTANT:", "")
            target = data["label"].replace("</s>", "")
        # BelleGroup/train_0.5M_CN
        # LLMs/Alpaca-ShareGPT
        # yahma/alpaca-cleaned
        # QingyiSi/Alpaca-CoT
        elif exist_and_not_none(data, "instruction") and exist_and_not_none(data, "output"):
            action_1 = data["input"].split("Action 1:\n[")[1].split("]\n\n### Action 2:\n[")[0]
            action_1_str = ''
            for token in action_1.split(' '):
                action_1_str += f"<Motion {token}> "
            action_1_str = action_1_str[:-1]        

            action_2 = data["input"].split("]\n\n### Action 2:\n[")[1].split("]")[0]
            action_2_str = ''
            for token in action_2.split(' '):
                action_2_str += f"<Motion {token}> "
            action_2_str = action_2_str[:-1]        

            input = "Action 1:\n[" + action_1_str + "]\n\n### Action 2:\n[" + action_2_str + "]"
            prompt = (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{data['instruction']}\n\n### Input:\n{input}\n\n### Response:\n"
            )

            target = data["output"]
        # Open-Orca/OpenOrca
        elif exist_and_not_none(data, "system_prompt") and exist_and_not_none(data, "response"):
            prompt = data["system_prompt"] + "\n" + data["question"]
            target = data["response"]
        # crumb/gpt4all-clean
        # nomic-ai/gpt4all-j-prompt-generations
        elif exist_and_not_none(data, "prompt") and exist_and_not_none(data, "response"):
            prompt = data["prompt"]
            target = data["response"]
        # EleutherAI/pile [pretrain !!!]
        elif exist_and_not_none(data, "text") and exist_and_not_none(data, "meta"):
            assert input_template is None  # pretrain_mode
            prompt = ""
            target = data["text"]
        # for batch_inference.py
        elif exist_and_not_none(data, "input") and exist_and_not_none(data, "output"):
            prompt = data["input"]
            target = data["output"]
            input_template = None
        else:
            raise ValueError("Unknown SFT dataset")

    # input template
    if input_template:
        prompt = input_template.format(prompt)
    return prompt, target


class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template="Human: {}\nAssistant: ",
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)

        self.prompts = []
        self.input_motions = []
        self.output_motions = []
        self.lengths = []
        self.max_motion_length = 196                                                                                                                

        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
            if type(data) == str:
                prompt, _ = preprocess_data(data, self.input_template, None, None)
                self.prompts.append(prompt)
            else:
                prompt, _ = preprocess_data(data, self.input_template, None, None)
                self.prompts.append(prompt)
                #self.input_motions.append(data['input_motion'])
                #self.output_motions.append(data['output_motion'])
                #s#zelf.lengths.append(data['length'])


        
        self.mean = np.load('/media/xuzhao/Partition5/qyzheng/qihang/Data/olddata/humanml3d/Mean.npy')
        self.std = np.load('/media/xuzhao/Partition5/qyzheng/qihang/Data/olddata/humanml3d/Std.npy')

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        if len(self.input_motions) != 0:
            input_motion = self.input_motions[idx]
            output_motions = self.output_motions[idx]

            if len(input_motion) < 196:
                input_motion = np.concatenate((input_motion, np.zeros((196 - len(input_motion), input_motion.shape[1]))))
            
            if len(output_motions) < 196:
                output_motions = np.concatenate((output_motions, np.zeros((196 - len(output_motions), output_motions.shape[1]))))


            return self.prompts[idx], input_motion, output_motions, self.lengths[idx]
        #else:
        #    return ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        #    "You are a helpful assistant.<|eot_id|>\n"  # The system prompt is optional
        #    "<|start_header_id|>user<|end_header_id|>\n\n"
        #    f"{self.prompts[idx]}<|eot_id|>\n")
            #"<|start_header_id|>assistant<|end_header_id|>\n\n")
        else:
            return self.prompts[idx]
    def inv_transform(self, data):
        return data * self.std + self.mean





class ETFTPromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template="Human: {}\nAssistant: ",
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)

        self.prompts = []
        self.targets = []
        self.input_motions = []
        self.output_motions = []
        self.lengths = []
        self.max_motion_length = 196                                                                                                                

        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
            if type(data) == str:
                prompt, target = preprocess_data_etft(data, self.input_template, None, None)
                self.prompts.append(prompt)
                self.targets.append(target)
            else:
                prompt, target = preprocess_data_etft(data, self.input_template, None, None)
                self.prompts.append(prompt)
                self.targets.append(target)

                #self.input_motions.append(data['input_motion'])
                #self.output_motions.append(data['output_motion'])
                #s#zelf.lengths.append(data['length'])


        
        self.mean = np.load('/media/xuzhao/Partition5/qyzheng/qihang/Data/olddata/humanml3d/Mean.npy')
        self.std = np.load('/media/xuzhao/Partition5/qyzheng/qihang/Data/olddata/humanml3d/Std.npy')

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        if len(self.input_motions) != 0:
            input_motion = self.input_motions[idx]
            output_motions = self.output_motions[idx]

            if len(input_motion) < 196:
                input_motion = np.concatenate((input_motion, np.zeros((196 - len(input_motion), input_motion.shape[1]))))
            
            if len(output_motions) < 196:
                output_motions = np.concatenate((output_motions, np.zeros((196 - len(output_motions), output_motions.shape[1]))))


            return self.prompts[idx], input_motion, output_motions, self.lengths[idx]
        #else:
        #    return ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        #    "You are a helpful assistant.<|eot_id|>\n"  # The system prompt is optional
        #    "<|start_header_id|>user<|end_header_id|>\n\n"
        #    f"{self.prompts[idx]}<|eot_id|>\n")
            #"<|start_header_id|>assistant<|end_header_id|>\n\n")
        else:
            #print(self.prompts[idx] + self.targets[idx] + " " + self.tokenizer.eos_token)
            ##exit()
            return self.prompts[idx] + 'The person'
    def inv_transform(self, data):
        return data * self.std + self.mean
