from transformers import BertModel, BertConfig
from transformers import SwinConfig, SwinModel
import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm


def seperate(last_output, a, answer_targets, q_num, dataset_name, qids=None):
    indexs_open = []
    indexs_close = []
    k = 0
    for i in range(len(answer_targets)):
        for j in range(q_num[i]):
            if answer_targets[i][j]==0:
                indexs_close.append(k)
            else:
                indexs_open.append(k)
            k = k + 1
    if dataset_name == "SLAKE":
        if qids != None:
            return last_output[indexs_open, :], last_output[indexs_close, :], \
                   a[indexs_open, 39:267], a[indexs_close, :39], \
                   qids[indexs_open], qids[indexs_close]
        return last_output[indexs_open, :], last_output[indexs_close, :], \
               a[indexs_open, 39:267], a[indexs_close, :39]
    else:
        if qids != None:
            return last_output[indexs_open, :], last_output[indexs_close, :], \
                   a[indexs_open, 56:487], a[indexs_close, :56], \
                   qids[indexs_open], qids[indexs_close]
        return last_output[indexs_open, :], last_output[indexs_close, :], \
               a[indexs_open, 56:487], a[indexs_close, :56]


class model(nn.Module):

    def __init__(self, args):
        super(model, self).__init__()
        self.device = torch.device("cuda:" + str(0) if 0 >= 0 else "cpu")
        self.dataset_name = args.dataset_name

        self.weight = nn.Parameter(torch.rand(1, 1))

        self.bert_config = BertConfig.from_pretrained("./bert-base-uncased", output_hidden_states=True, output_attentions=True)
        self.bert = BertModel.from_pretrained("./bert-base-uncased", config =self.bert_config)

        self.swin = SwinModel.from_pretrained("./microsoft/swin-small-patch4-window7-224")

        self.transformer = nn.Transformer(d_model=768, nhead=8, num_encoder_layers=3, num_decoder_layers=3, batch_first=True)
        if self.dataset_name == "SLAKE":
            self.classifier_open = nn.Linear(768, 228)
            self.classifier_close = nn.Linear(768, 39)
        else:
            self.classifier_open = nn.Linear(768, 431)
            self.classifier_close = nn.Linear(768, 56)

    def forward(self, image, question, mask_pos, answer_targets, a, q_num, qids=None):
        image = torch.squeeze(image, 1)
        v_emb = self.swin(pixel_values=image).last_hidden_state

        bert_emb = self.bert(input_ids=question[0], attention_mask=question[1])
        q_emb = bert_emb[0]

        last_output = self.transformer(v_emb, q_emb)
        # last_output = self.transformer(torch.cat((q_emb, v_emb), 1))
        last_output_clean = torch.FloatTensor(torch.sum(q_num, 0), last_output.size(2)).to(self.device)


        # last_output = q_emb + self.weight * last_output
        last_output = last_output


        k = 0
        for i in range(len(q_num)):
            last_mask_pos = 1
            for j in range(q_num[i]):
                last_output_clean[k] = last_output[i][mask_pos[i][j], :]
                k = k + 1

        a_clean = torch.FloatTensor(torch.sum(q_num, 0), a.size(2)).to(self.device)
        k = 0
        for i in range(len(q_num)):
            for j in range(q_num[i]):
                a_clean[k] = a[i][j]
                k = k + 1

        if qids != None:
            qids_clean = torch.FloatTensor(torch.sum(q_num, 0), 1).to(self.device)
            k = 0
            for i in range(len(q_num)):
                for j in range(q_num[i]):
                    qids_clean[k] = qids[i][j]
                    k = k + 1

        if qids != None:
            last_output_open, last_output_close, a_open, a_close, qid_open, qid_close = seperate(last_output_clean, a_clean, answer_targets, q_num, self.dataset_name, qids_clean)
        else:
            last_output_open, last_output_close, a_open, a_close = seperate(last_output_clean, a_clean, answer_targets, q_num, self.dataset_name)

        if self.dataset_name == "SLAKE":
            out_open = self.classifier_open(last_output_open)
            out_close = self.classifier_close(last_output_close)
            if qids != None:
                return out_open, out_close, a_open, a_close, qid_open, qid_close
            return out_open, out_close, a_open, a_close
        else:
            out_open = self.classifier_open(last_output_open)
            out_close = self.classifier_close(last_output_close)
            if qids != None:
                return out_open, out_close, a_open, a_close, qid_open, qid_close
            return out_open, out_close, a_open, a_close