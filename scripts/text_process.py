import argparse
import spacy

from tqdm import tqdm
import codecs as cs
from os.path import join as pjoin

import joblib
import jsonlines


nlp = spacy.load('en_core_web_sm')
def process_text(sentence):
    sentence = sentence.replace('-', '')
    doc = nlp(sentence)
    word_list = []
    pos_list = []
    for token in doc:
        word = token.text
        if not word.isalpha():
            continue
        if (token.pos_ == 'NOUN' or token.pos_ == 'VERB') and (word != 'left'):
            word_list.append(token.lemma_)
        else:
            word_list.append(word)
        pos_list.append(token.pos_)
    return word_list, pos_list

def process_humanml3d(input_file, output_file):
    data_list = []
    
    with jsonlines.open(input_file) as f:
        for line in f:
            data_list.append(line)


    output_data = []
    import tqdm
    for data in tqdm.tqdm(data_list):
        if True:
            line = data['output']
            line = line.strip().replace('\n', '')[:-1]
            if len(line) == 0:
                output_data.append(output_data[-1])
                continue
            if line:
                caption = line.strip()
                word_list, pose_list = process_text(caption)
                tokens = ' '.join(['%s/%s'%(word_list[i], pose_list[i]) for i in range(len(word_list))])
                output_data.append({'tokens':'%s#%s#%s#%s\n'%(caption, tokens, 0.0, 0.0)})

    joblib.dump(output_data, output_file)

def process_kitml(corpus):
    text_save_path = './dataset/kit_mocap_dataset/texts'
    desc_all = corpus
    for i in tqdm(range(len(desc_all))):
        caption = desc_all.iloc[i]['desc']
        start = 0.0
        end = 0.0
        name = desc_all.iloc[i]['data_id']
        word_list, pose_list = process_text(caption)
        tokens = ' '.join(['%s/%s' % (word_list[i], pose_list[i]) for i in range(len(word_list))])
        with cs.open(pjoin(text_save_path, name + '.txt'), 'a+') as f:
            f.write('%s#%s#%s#%s\n' % (caption, tokens, start, end))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()


    process_humanml3d(args.input_file, args.output_file)