import os
import _pickle as cPickle
import torch
from torch.distributions import Categorical


result_path = './saved_models/2024Jan26-121420_rebuttal_SLAKE_No_QA/estimate_results.pkl'
load_estimated_result_0 = cPickle.load(open(os.path.join(result_path), 'rb'))

result_path = './saved_models/2024Jan30-125314_rebuttal_SLAKE_real/estimate_results.pkl' # './saved_models/2024Jan26-115211_rebuttal_SLAKE_30%_error/estimate_results.pkl'
load_estimated_result_p = cPickle.load(open(os.path.join(result_path), 'rb'))

mean_info_gain = 0

for k in load_estimated_result_p.keys():
    preds_0 = torch.softmax(load_estimated_result_0[k], dim=0)
    preds_p = torch.softmax(load_estimated_result_p[k], dim=0)
    info_gain = Categorical(preds_p).entropy() - Categorical(preds_0).entropy()
    mean_info_gain = mean_info_gain + info_gain

print(mean_info_gain / len(load_estimated_result_0))


result_path = os.path.join('./saved_models', 'flitered_results_ereal.txt')
with open(result_path, "w", encoding='utf-8') as f:
    for k, v in load_estimated_result_p.items():
        preds_0 = torch.softmax(load_estimated_result_0[k], dim=0)
        preds_p = torch.softmax(load_estimated_result_p[k], dim=0)
        info_gain = Categorical(preds_p).entropy() - Categorical(preds_0).entropy()
        if info_gain > 1e-4:
            v = load_estimated_result_0[k]
        else:
            v = load_estimated_result_p[k]
        v = (load_estimated_result_0[k] + load_estimated_result_p[k]) / 2
        text = str(k) + "," + str(torch.max(v, 0)[1].data) + "," + str(torch.max(torch.softmax(v, 0), 0)[0].data) + "\n"
        f.write(text)
f.close()