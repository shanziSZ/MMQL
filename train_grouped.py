import os
import time
import utils
import statistics
import _pickle as cPickle
from datetime import datetime
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from dataset_SLAKE_group import SLAKEGroupFeatureDataset
from dataset_RAD_group import RADGroupFeatureDataset


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).to(logits.device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def get_time_stamp():
    ct = time.time()
    local_time = time.localtime(ct)
    data_head = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
    data_secs = (ct - int(ct)) * 1000
    time_stamp = "%s.%03d" % (data_head, data_secs)
    return time_stamp


# Train phase
def train(args, model, train_loader, eval_loader, s_opt=None, s_epoch=0):
    device = args.device
    model = model.to(device)
    utils.create_dir(args.output)
    run_timestamp = datetime.now().strftime("%Y%b%d-%H%M%S")
    ckpt_path = os.path.join(args.output, run_timestamp)
    utils.create_dir(ckpt_path)
    logger = utils.Logger(os.path.join(ckpt_path, 'medVQA.log')).get_logger()
    # logger.info(">>>The net is:")
    # logger.info(model)
    logger.info(">>>The args is:")
    logger.info(args.__repr__())

    total = sum([param.nelement() for param in model.parameters()])
    print('Number of parameter: % .4fM' % (total / 1e6))
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    total_time = 0

    # Optim
    optim = torch.optim.AdamW([
        {'params': [p for n, p in list(model.bert.named_parameters())], 'lr': 5e-5},

        {'params': [p for n, p in list(model.swin.named_parameters())], 'lr': 1e-4},

        {'params': [p for n, p in list(model.transformer.named_parameters())], 'lr': 1e-4},

        {'params': [model.weight]},

        {'params': [p for n, p in list(model.classifier_open.named_parameters())]},
        {'params': [p for n, p in list(model.classifier_close.named_parameters())]},
        ],
        lr=1e-4
    )

    # Scheduler learning rate
    lr_decay = lr_scheduler.CosineAnnealingLR(optim,T_max=args.epochs)
    criterion = nn.CrossEntropyLoss().to(device)

    best_eval_score = 0
    best_correct_num = 0
    best_open_score = 0
    best_close_score = 0
    best_answer = {}

    # Epoch passing in training phase
    for epoch in range(s_epoch, args.epochs):

        if args.dataset_name == "SLAKE":
            train_dataset = SLAKEGroupFeatureDataset('train', args, dataroot='./data-SLAKE')
            # train_dataset = SLAKEStarGroupFeatureDataset('train', args, dataroot='./data-SLAKE')
            train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=0, drop_last=False,pin_memory=True)
        else:
            train_dataset = RADGroupFeatureDataset('train', args, dataroot='../data-RAD')
            train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=0, drop_last=False, pin_memory=True)

        total_loss = 0
        train_score = 0
        number = 0
        model.train()

        # Predicting and computing score
        for ii, (v, q_ids, q_mask, a, answer_types, answer_targets, mask_pos, q_num, qids) in enumerate(train_loader):
            optim.zero_grad()

            v = v[0].to(device)
            q_ids = q_ids.to(device)
            q_mask = q_mask.to(device)
            a = a.to(device)
            for i in range(len(mask_pos)):
                mask_pos[i] = torch.unsqueeze(mask_pos[i], 0)
                answer_targets[i] = torch.unsqueeze(answer_targets[i], 0)
            mask_pos = torch.cat(mask_pos, 0)
            mask_pos = torch.transpose(mask_pos, dim0=0, dim1=1)
            answer_targets = torch.cat(answer_targets, 0)
            answer_targets = torch.transpose(answer_targets, dim0=0, dim1=1)

            preds_open, preds_close, a_open, a_close = model(v, [q_ids, q_mask], mask_pos, answer_targets, a, q_num)

            if args.dataset_name == "SLAKE":
                loss = criterion(preds_open.float(), a_open) + criterion(preds_close.float(), a_close)
            else:
                loss = criterion(preds_open.float(), a_open) + criterion(preds_close.float(), a_close)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optim.step()

            if args.dataset_name == "SLAKE":
                score = compute_score_with_logits(preds_open, a_open.data).sum() + compute_score_with_logits(preds_close, a_close.data).sum()
            else:
                score = compute_score_with_logits(preds_open, a_open.data).sum() + compute_score_with_logits(preds_close, a_close.data).sum()
            train_score += score
            total_loss += loss.item()
            if args.dataset_name == "SLAKE":
                number += preds_open.shape[0] + preds_close.shape[0]
            else:
                number += preds_open.shape[0] + preds_close.shape[0]

        lr_decay.step()

        total_loss /= number
        train_score = 100 * train_score / number
        logger.info('-------[Epoch]:{}-------'.format(epoch))
        logger.info('[Train] Loss:{:.6f} , Train_Acc:{:.6f}%'.format(total_loss, train_score))

        # Evaluation
        model.eval()
        total = 0
        eval_loss = 0
        eval_score = 0
        total_loss = 0
        num_open = 0
        num_close = 0
        if args.dataset_name != "SLAKE":
            num_open += 0.00000001
            num_close += 0.00000001
        correct_open = 0
        correct_close = 0
        epoch_answer = {}
        if eval_loader is not None:

            starter.record()

            for ii, (v, q_ids, q_mask, a, answer_types, answer_targets, mask_pos, q_num, qids) in enumerate(eval_loader):
                optim.zero_grad()

                v = v[0].to(device)
                q_ids = q_ids.to(device)
                q_mask = q_mask.to(device)
                a = a.to(device)
                for i in range(len(mask_pos)):
                    mask_pos[i] = torch.unsqueeze(mask_pos[i], 0)
                    answer_targets[i] = torch.unsqueeze(answer_targets[i], 0)
                    qids[i] = torch.unsqueeze(qids[i], 0)
                mask_pos = torch.cat(mask_pos, 0)
                mask_pos = torch.transpose(mask_pos, dim0=0, dim1=1)
                answer_targets = torch.cat(answer_targets, 0)
                answer_targets = torch.transpose(answer_targets, dim0=0, dim1=1)
                qids = torch.cat(qids, 0)
                qids = torch.transpose(qids, dim0=0, dim1=1)

                preds_open, preds_close, a_open, a_close, qid_open, qid_close = model(v, [q_ids, q_mask], mask_pos, answer_targets, a, q_num, qids)

                # compute the acc for open and close
                if args.dataset_name == "SLAKE":
                    score = compute_score_with_logits(preds_open, a_open.data).sum() + compute_score_with_logits(preds_close, a_close.data).sum()
                else:
                    score = compute_score_with_logits(preds_open, a_open.data).sum() + compute_score_with_logits(preds_close, a_close.data).sum()
                eval_score += score

                if args.dataset_name == "SLAKE":
                    total += preds_open.shape[0] + preds_close.shape[0]
                    num_open += preds_open.shape[0]
                    num_close += preds_close.shape[0]
                    correct_open += compute_score_with_logits(preds_open, a_open.data).sum()
                    correct_close += compute_score_with_logits(preds_close, a_close.data).sum()
                    preds_open_labels = torch.max(preds_open, 1)[1].data
                    qid_open = qid_open.data
                    utils.assert_eq(len(preds_open_labels), len(qid_open))
                    for jj in range(len(qid_open)):
                        epoch_answer[qid_open[jj]] = preds_open_labels[jj]
                    preds_close_labels = torch.max(preds_close, 1)[1].data
                    qid_close = qid_close.data
                    utils.assert_eq(len(preds_close_labels), len(qid_close))
                    for jj in range(len(qid_close)):
                        epoch_answer[qid_close[jj]] = preds_close_labels[jj]
                else:
                    total += preds_open.shape[0]
                    total += preds_close.shape[0]
                    num_open += preds_open.shape[0]
                    num_close += preds_close.shape[0]
                    correct_open += compute_score_with_logits(preds_open, a_open.data).sum()
                    correct_close += compute_score_with_logits(preds_close, a_close.data).sum()

                    preds_open_labels = torch.max(preds_open, 1)[1].data
                    qid_open = qid_open.data
                    utils.assert_eq(len(preds_open_labels), len(qid_open))
                    for jj in range(len(qid_open)):
                        epoch_answer[qid_open[jj]] = preds_open_labels[jj]
                    preds_close_labels = torch.max(preds_close, 1)[1].data
                    qid_close = qid_close.data
                    utils.assert_eq(len(preds_close_labels), len(qid_close))
                    for jj in range(len(qid_close)):
                        epoch_answer[qid_close[jj]] = preds_close_labels[jj]

            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            if epoch > 0:
                total_time = total_time + curr_time
            # print("The batch time is: ", curr_time, epoch)

            # Save results
            result_path = os.path.join(ckpt_path, str(epoch) + '_predict_results.txt')
            with open(result_path, "w", encoding='utf-8') as f:
                for k, v in best_answer.items():
                    text = str(k) + "," + str(v) + "\n"
                    f.write(text)
            f.close()

            total_loss /= total
            if 100 * eval_score / total > best_eval_score:
                best_eval_score = 100 * eval_score / total
                best_correct_num = eval_score
                best_open_score = correct_open / num_open * 100
                best_close_score = correct_close / num_close * 100
                model_path = os.path.join(ckpt_path, str(epoch) + '_model.pth')
                logger.info(model_path)
                best_answer = epoch_answer
                utils.save_model(model_path, model, epoch, optim)
            logger.info('[Result] Loss:{:.6f}, The acc is {:.6f}%, The correct num is {}, '
                        'the acc of open question is {:.6f}%, '
                        'the acc of close question is {:.6f}%'.format(eval_loss,
                                                                      100 * eval_score / total,
                                                                      eval_score,
                                                                      correct_open / num_open * 100,
                                                                      correct_close / num_close * 100))
    print("The average batch time is: ", total_time / 199, total_time)

    logger.info('The best acc is {:.6f}%, the best correct num is {:.6f}%, '
                'the best acc of open question is {:.6f}%, '
                'the best acc of close question is {:.6f}%'.format(best_eval_score, best_correct_num, best_open_score, best_close_score))

    result_path = os.path.join(ckpt_path, 'predict_results.txt')
    # 打开文件
    with open(result_path, "w", encoding='utf-8') as f:
        for k,v in best_answer.items():
            text = str(k) + "," + str(v) + "\n"
            f.write(text)
    print(len(best_answer))


def test(args, model, eval_loader):
    device = args.device
    model = model.to(device)
    utils.create_dir(args.output)
    run_timestamp = datetime.now().strftime("%Y%b%d-%H%M%S")
    ckpt_path = os.path.join(args.output, run_timestamp)
    utils.create_dir(ckpt_path)
    logger = utils.Logger(os.path.join(ckpt_path, 'medVQA.log')).get_logger()

    result_list = []
    estimated_preds = {}

    for epoch in range(100):

        print(epoch)

        if args.dataset_name == "SLAKE":
            eval_dataset = SLAKEGroupFeatureDataset('test', args, dataroot='./data-SLAKE')
            eval_loader = DataLoader(eval_dataset, 1, shuffle=False, num_workers=0, drop_last=False,pin_memory=True)
        else:
            eval_dataset = RADGroupFeatureDataset('test', args, dataroot='./data-RAD')
            eval_loader = DataLoader(eval_dataset, args.batch_size, shuffle=True, num_workers=0, drop_last=False, pin_memory=True)

        # Evaluation
        model.eval()
        total = 0
        eval_loss = 0
        eval_score = 0
        num_open = 0
        num_close = 0
        if args.dataset_name != "SLAKE":
            num_open += 0.00000001
            num_close += 0.00000001
        correct_open = 0
        correct_close = 0
        epoch_answer = {}
        epoch_prob = {}

        for ii, (v, q_ids, q_mask, a, answer_types, answer_targets, mask_pos, q_num, qids) in enumerate(eval_loader):
            v = v[0].to(device)
            q_ids = q_ids.to(device)
            q_mask = q_mask.to(device)
            a = a.to(device)
            for i in range(len(mask_pos)):
                mask_pos[i] = torch.unsqueeze(mask_pos[i], 0)
                answer_targets[i] = torch.unsqueeze(answer_targets[i], 0)
                qids[i] = torch.unsqueeze(qids[i], 0)
            mask_pos = torch.cat(mask_pos, 0)
            mask_pos = torch.transpose(mask_pos, dim0=0, dim1=1)
            answer_targets = torch.cat(answer_targets, 0)
            answer_targets = torch.transpose(answer_targets, dim0=0, dim1=1)
            qids = torch.cat(qids, 0)
            qids = torch.transpose(qids, dim0=0, dim1=1)

            preds_open, preds_close, a_open, a_close, qid_open, qid_close = model(v, [q_ids, q_mask], mask_pos, answer_targets, a, q_num, qids)

            # compute the acc for open and close
            if args.dataset_name == "SLAKE":
                score = compute_score_with_logits(preds_open, a_open.data).sum() + compute_score_with_logits(preds_close, a_close.data).sum()
            else:
                score = compute_score_with_logits(preds_open, a_open.data).sum() + compute_score_with_logits(preds_close, a_close.data).sum()
            eval_score += score

            if args.dataset_name == "SLAKE":
                total += preds_open.shape[0] + preds_close.shape[0]
                num_open += preds_open.shape[0]
                num_close += preds_close.shape[0]
                correct_open += compute_score_with_logits(preds_open, a_open.data).sum()
                correct_close += compute_score_with_logits(preds_close, a_close.data).sum()
                preds_open_labels = torch.max(preds_open, 1)[1].data
                qid_open = qid_open.data
                utils.assert_eq(len(preds_open_labels), len(qid_open))
                for jj in range(len(qid_open)):
                    epoch_answer[str(qid_open[jj])] = preds_open_labels[jj]
                    epoch_prob[str(qid_open[jj])] = torch.max(torch.softmax(preds_open, 1), 1)[0].data[jj]
                    if str(qid_open[jj]) not in estimated_preds:
                        estimated_preds[str(qid_open[jj])] = torch.zeros(preds_open.size()[1])
                    estimated_preds[str(qid_open[jj])] = estimated_preds[str(qid_open[jj])] + preds_open[jj].detach().cpu()
                preds_close_labels = torch.max(preds_close, 1)[1].data
                qid_close = qid_close.data
                utils.assert_eq(len(preds_close_labels), len(qid_close))
                for jj in range(len(qid_close)):
                    epoch_answer[str(qid_close[jj])] = preds_close_labels[jj]
                    epoch_prob[str(qid_close[jj])] = torch.max(torch.softmax(preds_close, 1), 1)[0].data[jj]
                    if str(qid_close[jj]) not in estimated_preds:
                        estimated_preds[str(qid_close[jj])] = torch.zeros(preds_close.size()[1])
                    estimated_preds[str(qid_close[jj])] = estimated_preds[str(qid_close[jj])] + preds_close[jj].detach().cpu()
            else:
                total += preds_open.shape[0]
                total += preds_close.shape[0]
                num_open += preds_open.shape[0]
                num_close += preds_close.shape[0]
                correct_open += compute_score_with_logits(preds_open, a_open.data).sum()
                correct_close += compute_score_with_logits(preds_close, a_close.data).sum()

                preds_open_labels = torch.max(preds_open, 1)[1].data
                qid_open = qid_open.data
                utils.assert_eq(len(preds_open_labels), len(qid_open))
                for jj in range(len(qid_open)):
                    epoch_answer[str(qid_open[jj])] = preds_open_labels[jj]
                    epoch_prob[str(qid_open[jj])] = torch.max(torch.softmax(preds_open, 1), 1)[0].data[jj]
                preds_close_labels = torch.max(preds_close, 1)[1].data
                qid_close = qid_close.data
                utils.assert_eq(len(preds_close_labels), len(qid_close))
                for jj in range(len(qid_close)):
                    epoch_answer[str(qid_close[jj])] = preds_close_labels[jj]
                    epoch_prob[str(qid_close[jj])] = torch.max(torch.softmax(preds_close, 1), 1)[0].data[jj]

        acc = 100 * eval_score / total
        acc = acc.cpu().numpy().tolist()
        result_list.append(acc)

        if len(result_list) == 1 or acc > max(result_list):
            result_path = os.path.join(ckpt_path, 'test_results.txt')
            # Save results
            with open(result_path, "w", encoding='utf-8') as f:
                for k, v in epoch_answer.items():
                    text = str(k) + "," + str(epoch_answer[k]) + "," + str(epoch_prob[k]) + "\n"
                    f.write(text)
            f.close()



        logger.info('[Result] Loss:{:.6f}, The acc is {:.6f}%, The correct num is {}, '
                    'the acc of open question is {:.6f}%, '
                    'the acc of close question is {:.6f}%'.format(eval_loss,
                                                                  100 * eval_score / total,
                                                                  eval_score,
                                                                  correct_open / num_open * 100,
                                                                  correct_close / num_close * 100))

    print(result_list)
    std_dev = statistics.stdev(result_list)
    print("Std: ", std_dev)
    print("Avg: ", sum(result_list) / 100)

    result_path = os.path.join(ckpt_path, 'estimate_results.pkl')
    cPickle.dump(estimated_preds, open(result_path, 'wb'))
    # load_estimated_result = cPickle.load(open(os.path.join(result_path), 'rb'))
    # print(load_estimated_result)
