from __future__ import print_function
import os
import json
import copy
import random
import warnings
import numpy as np
from tqdm import tqdm
from PIL import Image
import _pickle as cPickle
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from transformers import AutoFeatureExtractor

import utils


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
COUNTING_ONLY = False


def is_howmany(q, a, label2ans):
    if 'how many' in q.lower() or \
            ('number of' in q.lower() and 'number of the' not in q.lower()) or \
            'amount of' in q.lower() or \
            'count of' in q.lower():
        if a is None or answer_filter(a, label2ans):
            return True
        else:
            return False
    else:
        return False


def answer_filter(answers, label2ans, max_num=10):
    for ans in answers['labels']:
        if label2ans[ans].isdigit() and max_num >= int(label2ans[ans]):
            return True
    return False


def _create_entry(img, data, answer):
    if None != answer:
        answer.pop('image_name')
        answer.pop('qid')
    entry = {
        'qid': data['qid'],
        'image_name': data['image_name'],
        'image': img,
        'question': data['question'],
        'answer': answer,
        'answer_type': data['answer_type'],
        'question_type': data['question_type'],
        'phrase_type': data['phrase_type'],
        'image_organ': data['image_organ']}
    return entry


def is_json(myjson):
    try:
        json_object = json.loads(myjson)
    except ValueError:
        return False
    return True


def _load_dataset(dataroot, name, img_id2val, label2ans):
    """Load entries
    img2id: dict {img -> id} id can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val', 'test'
    """
    data_path = os.path.join(dataroot, name + 'set.json')
    samples = json.load(open(data_path))
    samples = sorted(samples, key=lambda x: x['qid'])

    answer_path = os.path.join(dataroot, 'cache', '%s_openclose_target.pkl' % name)
    answers = cPickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['qid'])

    utils.assert_eq(len(samples), len(answers))
    entries = {}
    for sample, answer in zip(samples, answers):
        utils.assert_eq(sample['qid'], answer['qid'])
        utils.assert_eq(sample['image_name'], answer['image_name'])
        img_id = sample['image_name']
        if not COUNTING_ONLY or is_howmany(sample['question'], answer, label2ans):
            entry = _create_entry(img_id2val[img_id], sample, answer)
            qid = entry['qid']
            entries[qid] = entry

    return entries


class RADGroupFeatureDataset(Dataset):
    def __init__(self, name, args, dataroot='data'):
        super(RADGroupFeatureDataset, self).__init__()
        self.name = name
        self.args = args
        self.dataroot = dataroot
        assert name in ['train', 'test']
        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.image2vit = cPickle.load(open("./data-RAD/ViTImage2idx.pkl", 'rb'))
        self.num_total_candidates = 487  # 56 431

        # Close & open
        self.label2close = cPickle.load(open(os.path.join(dataroot, 'cache', 'close_label2ans.pkl'), 'rb'))
        self.label2open = cPickle.load(open(os.path.join(dataroot, 'cache', 'open_label2ans.pkl'), 'rb'))
        self.num_open_candidates = len(self.label2open)
        self.num_close_candidates = len(self.label2close)


        self.img_id2idx = json.load(open(os.path.join(dataroot, 'imgid2idx.json')))

        # Load dataset
        self.entries_train = _load_dataset(dataroot, 'train', self.img_id2idx, self.label2ans)
        self.entries_test = _load_dataset(dataroot, 'test', self.img_id2idx, self.label2ans)

        # Group samples
        self.test_qid = []
        self.groupedEntries = self.map_entry(self.entries_train, self.entries_test)
        self.imageID_map = self.map_image2id()

        # Get visual feature extractor
        self.swin_feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-small-patch4-window7-224")
        self.swim_image_features = self.get_swim_images()

        # Tokenize & Tensorize
        self.tokenize()
        self.tensorize()


    def get_swim_images(self):
        image_feature_pair = {}
        for k, entry in self.groupedEntries.items():
            image_name = k
            if not image_name in image_feature_pair:
                img_for_swin = Image.open(os.path.join(self.dataroot, "images", image_name)).convert('RGB')
                image_feature = self.swin_feature_extractor(images=img_for_swin, return_tensors="pt")['pixel_values']
                image_feature_pair[image_name] = image_feature
        return image_feature_pair


    def map_entry(self, entries_train, entries_test):
        entry_map = {}
        for k,v in entries_train.items():
            image_id = v['image_name']
            if image_id in entry_map:
                pass
            else:
                entry_map[image_id] = {}
                entry_map[image_id]['meta_data'] = []
            entry_map[image_id]['meta_data'].append(v)
        if self.name == "test":
            for k, v in entries_test.items():
                image_id = v['image_name']
                if image_id in entry_map:
                    pass
                else:
                    entry_map[image_id] = {}
                    entry_map[image_id]['meta_data'] = []
                entry_map[image_id]['meta_data'].append(v)
                self.test_qid.append(v['qid'])
        return entry_map
    
    
    def map_image2id(self):
        id_map = []
        for k in self.groupedEntries.keys():
            id_map.append(k)
        return id_map


    def tokenize(self):
        """Tokenizes the questions.
        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

        len_list = []

        for image_id in tqdm(self.groupedEntries.keys()):
            group_entries = self.groupedEntries[image_id]['meta_data']

            random.shuffle(group_entries)

            self.groupedEntries[image_id]['meta_data'] = group_entries

            questions = []
            answers = []
            qids = []
            open_close = []
            for entry in group_entries:
                    questions.append(entry['question'])
                    answers.append(entry['answer'])
                    qids.append(entry['qid'])
                    open_close.append(entry["answer_type"])

            answers_clean = []
            open_close_clean = []

            sentence = ""
            if self.name == "train":
                for i, question in enumerate(questions):
                    if random.random() < 0.3:
                        sentence = sentence + "Question: " + question + "Answer: [MASK]. "
                        answers_clean.append(answers[i])
                        open_close_clean.append(open_close[i])
                    else:
                        answers_str = self.label2ans[answers[i]['labels'][0]]
                        sentence = sentence + "Question: " + question + "Answer: " + answers_str + ". "
            else:
                for i, question in enumerate(questions):
                    if qids[i] in self.test_qid:
                        sentence = sentence + "Question: " + question + "Answer: [MASK]. "
                        answers_clean.append(answers[i])
                        open_close_clean.append(open_close[i])
                    else:
                        answers_str = self.label2ans[answers[i]['labels'][0]]
                        sentence = sentence + "Question: " + question + "Answer: " + answers_str + ". "

            tokens = tokenizer(sentence,
                               padding="max_length",
                               truncation=False,
                               add_special_tokens=True,
                               max_length=430)

            len_list.append(len(tokens["input_ids"]))
            self.groupedEntries[image_id]['q_token'] = tokens
            self.groupedEntries[image_id]['answers'] = answers_clean
            self.groupedEntries[image_id]['answer_type'] = open_close_clean
            mask_pos = []
            for mask_idx in range(len(self.groupedEntries[image_id]['q_token']["input_ids"])):
                if self.groupedEntries[image_id]['q_token']["input_ids"][mask_idx] == 103:
                    mask_pos.append(mask_idx)
            utils.assert_eq(len(mask_pos), len(answers_clean))
            utils.assert_eq(len(mask_pos), len(open_close_clean))
            self.groupedEntries[image_id]['mask_pos'] = mask_pos

        len_list.sort()


    def tensorize(self):
        num_answers = []
        for image_id in tqdm(self.groupedEntries.keys()):
            self.groupedEntries[image_id]['q_token']["input_ids"] = np.array(self.groupedEntries[image_id]['q_token']["input_ids"])
            self.groupedEntries[image_id]['q_token']["attention_mask"] = np.array(self.groupedEntries[image_id]['q_token']["attention_mask"])

            answers = self.groupedEntries[image_id]['answers']
            num_answers.append(len(answers))

            precessed_answers = []
            for i, answer in enumerate(answers):
                precessed_answer = {}
                if None != answer:
                    labels = np.array(answer['labels'])
                    scores = np.array(answer['scores'], dtype=np.float32)
                    if len(labels):
                        labels = torch.from_numpy(labels)
                        scores = torch.from_numpy(scores)
                        precessed_answer['labels'] = labels
                        precessed_answer['scores'] = scores
                    else:
                        precessed_answer['labels'] = None
                        precessed_answer['scores'] = None
                    precessed_answers.append(precessed_answer)
                self.groupedEntries[image_id]['answers'] = precessed_answers

            self.groupedEntries[image_id]['answers'] = precessed_answers
            utils.assert_eq(len(self.groupedEntries[image_id]['answers']), len(self.groupedEntries[image_id]['mask_pos']))
            utils.assert_eq(len(self.groupedEntries[image_id]['answers']), len(self.groupedEntries[image_id]['answer_type']))


    def __getitem__(self, index):
        image_name = self.imageID_map[index]
        image_id = image_name
        question_ids = self.groupedEntries[image_id]['q_token']["input_ids"]
        question_mask = self.groupedEntries[image_id]['q_token']["attention_mask"]
        answers = self.groupedEntries[image_id]['answers']
        answer_types = copy.deepcopy(self.groupedEntries[image_id]['answer_type'])
        mask_pos = copy.deepcopy(self.groupedEntries[image_id]['mask_pos'])
        qids = []
        for meta_entry in self.groupedEntries[image_id]['meta_data']:
            if meta_entry['qid'] in self.test_qid:
                    qids.append(meta_entry['qid'])
        if self.name == "test":
            utils.assert_eq(len(qids), len(answers))

        image_data = [0, 0, 0]
        image_data[0] = self.swim_image_features[image_name]

        answer_targets = []
        composed_targets = []
        for i in range(len(answers)):
            labels = answers[i]['labels']
            scores = answers[i]['scores']
            composed_target = torch.zeros(self.num_total_candidates)  # close + open

            answer_type = answer_types[i]
            if answer_type == 'CLOSED':
                answer_target = 0
            else:
                answer_target = 1

            if answer_target == 0:
                target = torch.zeros(self.num_close_candidates)
                if labels is not None:
                    target.scatter_(0, labels, scores)
                composed_target[:self.num_close_candidates] = target
            else:
                target = torch.zeros(self.num_open_candidates)
                if labels is not None:
                    target.scatter_(0, labels, scores)
                composed_target[self.num_close_candidates: self.num_total_candidates] = target
            answer_targets.append(answer_target)
            composed_target = torch.unsqueeze(composed_target, 0)
            composed_targets.append(composed_target)

        while len(answer_targets) < 24:
            composed_target = torch.zeros(1, self.num_total_candidates)
            answer_type = "PADDING"
            answer_target = -1
            mask_pos_ = -1
            if self.name == "test":
                qids.append(-1)
            composed_targets.append(composed_target)
            answer_types.append(answer_type)
            answer_targets.append(answer_target)
            mask_pos.append(mask_pos_)
        composed_targets = torch.cat(composed_targets, 0)

        return image_data, question_ids, question_mask, composed_targets, answer_types, answer_targets, mask_pos, len(answers), qids

    def __len__(self):
        return len(self.groupedEntries)


