# CigTime: Corrective Instruction Generation Through Inverse Motion Editing


[Qihang Fang](https://qhfang.github.io/)<sup> 1 </sup>, 
[Chengcheng Tang](https://scholar.google.com/citations?user=WbG27wQAAAAJ)<sup> 2 </sup>, 
[Bugra Tekin](https://btekin.github.io/)<sup> 2 </sup>, 
[Yanchao Yang](https://yanchaoyang.github.io/)<sup> 1 </sup>, 


<sup>1</sup>The University of Hong Kong, 
<sup>2</sup>Meta the Reality Lab

<p align="center">
  <a href='https://openreview.net/pdf?id=gktA1Qycj9'>
  <img src='https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow'>
  </a> 
  <a href='https://qhfang.github.io/projects/CigTime/index.html'>
  <img src='https://img.shields.io/badge/Project-Page-%23df5b46?style=flat&logo=Google%20chrome&logoColor=%23df5b46'></a> 
  <a href='https://github.com/qhFang/CigTime'>
  <img src='https://img.shields.io/badge/GitHub-Code-black?style=flat&logo=github&logoColor=white'></a> 
</p>


# Abstract

Recent advancements in models linking natural language with human motions have shown significant promise in motion generation and editing based on instructional text. Motivated by applications in sports coaching and motor skill learning, we investigate the inverse problem: generating corrective instructional text, leveraging motion editing and generation models. We introduce a novel approach that, given a user’s current motion (source) and the desired motion (target), generates text instructions to guide the user towards achieving the target motion. We leverage large language models to generate corrective texts and utilize existing motion generation and editing frameworks to compile datasets of triplets (source motion, target motion, and corrective text). Using this data, we propose a new motion-language model for generating corrective instructions. We present both qualitative and quantitative results across a diverse range of applications that largely improve upon baselines. Our approach demonstrates its effectiveness in instructional scenarios, offering text-based guidance to correct and enhance user performance.


### Set up the environment

```bash
git clone https://github.com/qhFang/CigTime.git
cd CigTime
pip install -r requirements.txt
```

For MDM, you need to install the requirments following [README](mdm/README.md). And you need to download the [T2M-GPT](https://github.com/Mael-zys/T2M-GPT) encoder checkpoint and [Humanml3d](https://github.com/EricGuo5513/HumanML3D) dataset. 


### Instruction Generation

We provide a simple example to generate corrective instructions. You can run it by

```bash
python script/instruction_generation.py --api_key YOUR_API_KEY --output_file YOUR_OUTPUT_FILE
```

### Motion Editing
You can edit motion sequences by

```bash
sh script/edit.sh MDM_MODEL_PATH  INSTRUCTION_FILE  OUTPUT_DIR 
```


### LLM Fine-Tuning
Before fine-tune the LLM, you need to tranfer the data to Alpeca style.
```bash
python script/prepare_alpeca.py  --destination_path  OUTPUT_DIR  --encoder_checkpoints_path T2M_ENCODER_PATH  --edit_results_path EDIT_DATA_PATH --dataset_dir HUMANML3D_DATASET_PATH
```

Then, you can fine-tune your LLM model by
```bash
python train/train.py  --pretrain  YOUR_LLM_MODEL  --save_path YOUR_MODEL_SAVE_PATH
```

### Inference
You can utilize the following code to generate corrective instruction for input motion pairs.
```bash
python train/train.py  --pretrain  YOUR_MODEL_SAVE_PATH  --output_path YOUR_RESULT_SAVE_PATH
```




## Acknowledgement
The code is on the basis of [Motion-Diffusion-model](https://github.com/GuyTevet/motion-diffusion-model), [Openrlhf](https://github.com/OpenRLHF/OpenRLHF), [T2M-GPT](https://github.com/Mael-zys/T2M-GPT), [lit-gpt](https://github.com/Lightning-AI/litgpt), and [HumanML3D](https://github.com/EricGuo5513/HumanML3D).


