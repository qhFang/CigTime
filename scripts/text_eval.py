## BLEU-4 ROUGE-L METEOR
import argparse


import tqdm
import clip
import torch
import joblib
import jsonlines

import numpy as np

from rouge import Rouge
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt-path', type=str, help='Path to ground truth file')
    parser.add_argument('--pred-path', type=str, help='Path to prediction file')
    args = parser.parse_args()

    smooth = SmoothingFunction()

    gt_lines = joblib.load(open(args.gt_path, 'rb'))
    pred_lines = []
    with jsonlines.open(args.pred_path) as f:
        for line in f:
            pred_lines.append(line)

    ### BLEU-4
    bleu4 = []
    for i in range(len(gt_lines)):
        gt_line = gt_lines[i]['output'].strip()
        pred_line = pred_lines[i]['output'].strip()
        bleu4.append(sentence_bleu([gt_line.split()], pred_line.split(), smoothing_function=smooth.method1))

    print('BLEU-4:', np.mean(bleu4))

    ### Rouge-L
    rougel = []
    rouge = Rouge()

    for i in range(len(gt_lines)):
        gt_line = gt_lines[i]['output'].strip()
        pred_line = pred_lines[i]['output'].strip()
        if len(pred_line)==0:
            rougel.append(0)
            continue
        scores = rouge.get_scores(pred_line, gt_line)
        rougel.append(scores[0]['rouge-1']['f'])

    print('ROUGE-1:', np.mean(rougel))

    ### METEOR
    meteor = []
    for i in range(len(gt_lines)):
        gt_line = gt_lines[i]['output'].strip()
        pred_line = pred_lines[i]['output'].strip()
        meteor.append(meteor_score([gt_line.split()], pred_line.split()))

    print('METEOR:', np.mean(meteor))



    ### CLIP Score
    clip_model, clip_preprocess = clip.load('ViT-B/32', device='cpu', jit=False)
    clip_model.eval()
    clip_model.cuda()

    distances = []
    for i in tqdm.tqdm(range(len(gt_lines))):
        gt_sample = [gt_lines[i]['output'].strip()]
        predict_sample = pred_lines[i]['output'].strip()


        gt_tokens = clip.tokenize(gt_sample, truncate=True).cuda()
        predict_tokens = clip.tokenize(predict_sample, truncate=True).cuda()

        gt_fea = clip_model.encode_text(gt_tokens).float()
        predict_fea = clip_model.encode_text(predict_tokens).float()


        distance = torch.nn.functional.cosine_similarity(gt_fea, predict_fea, dim=-1).mean()
        distances.append(distance.item())

    print('CLIP Score:', np.mean(distances))

