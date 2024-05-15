import os
from dataset_SLAKE_group import SLAKEGroupFeatureDataset
from dataset_RAD_group import RADGroupFeatureDataset
import argparse
import _pickle as cPickle

def parse_args():
    parser = argparse.ArgumentParser(description="MMQL")

    # GPU config
    parser.add_argument('--seed', type=int, default=717 , help='random seed for gpu.default:5')
    parser.add_argument('--gpu', type=int, default=0, help='use gpu device. default:0')

    # Model loading/saving
    parser.add_argument('--input', type=str, default=None, help='input file directory for continue training from stop one')
    parser.add_argument('--output', type=str, default='saved_models', help='save file directory')

    # Training testing or sampling Hyper-parameters
    parser.add_argument('--epochs', type=int, default=200, help='the number of epoches')
    # parser.add_argument('--lr', default=1e-5, type=float, metavar='lr', help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--update_freq', default='1', metavar='N', help='update parameters every n batches in an epoch')
    parser.add_argument('--print_interval', default=20, type=int, metavar='N', help='print per certain number of steps')
    parser.add_argument('--clip_norm', default=.25, type=float, metavar='NORM', help='clip threshold of gradients')

    # # Train
    parser.add_argument('--use_data', action='store_true', default=True, help='Using TDIUC dataset to train')
    parser.add_argument('--data_dir', type=str, help='RAD dir')

    # details
    parser.add_argument('--details', type=str, default='original ')

    # dataset name
    parser.add_argument('--dataset_name', type=str, default='SLAKE')

    args = parser.parse_args()
    return args


def CaseAccuracy(predict_results, test_dataset, dataset_name):

    answers = {}

    if dataset_name == "SLAKE":
        label2close = cPickle.load(open('./data-SLAKE/cache/close_ans2labels.pkl', 'rb'))
        label2open = cPickle.load(open('./data-SLAKE/cache/open_ans2labels.pkl', 'rb'))

        lines = open('./data-SLAKE/provided.txt', 'r').readlines()
        provided = lines[0].split(',')

    cases = test_dataset.groupedEntries
    case_number = len(test_dataset)
    correct_case = 0
    correct_questions = 0
    correct_open = 0
    correct_close = 0
    no_test_q = 0

    for k, v in cases.items():
        in_case_questions = v['meta_data']
        pred_list = []
        gt_list = []
        for question in in_case_questions:
            qid = question["qid"]

            if dataset_name == "RAD":
                if qid not in test_dataset.test_qid:
                    continue
                if str(qid) not in predict_results.keys():
                    print(qid)
            else:
                if str(qid) in provided:
                    continue

            answer = question['answer']
            answer_type = question['answer_type']
            label = "-1"
            if dataset_name == "RAD":
                if len(answer["labels"]):
                    if question["answer_type"] == "CLOSED":
                        label = str(answer["labels"][0])
                    else:
                        label = str(answer["labels"][0])
            else:
                if answer_type == 'OPEN':
                    label = str(test_dataset.label2open[answer])
                else:
                    label = str(test_dataset.label2close[answer])

            predict_label = str(predict_results[str(qid)])
            pred_list.append(predict_label)
            gt_list.append(label)
            # print(label, predict_label)
            if label != predict_label:
                pass
            else:
                correct_questions = correct_questions + 1
                if question['answer_type'] == "OPEN":
                    correct_open = correct_open + 1
                else:
                    correct_close = correct_close + 1

        if len(gt_list) == 0:
            no_test_q = no_test_q + 1

        if pred_list == gt_list:
            correct_case = correct_case + 1
        else:
            pass


    if dataset_name == "RAD":
        print('The case acc is {:.6f}%, '
              'the correct case num is {}, '
              'the question acc is {:.6f}%, '
              'the correct question is {}.'.
              format(correct_case / case_number * 100, correct_case,
                     100 * correct_questions / len(test_dataset.entries_test), correct_questions))
        print('The close acc is {:.6f}%, '
              'the open acc is {:.6f}%.'.
              format(correct_close / 272 * 100, correct_open / 179 * 100))
    else:
        print('The case acc is {:.6f}%, '
              'the correct case num is {}, '
              'the question acc is {:.6f}%.'.
              format(correct_case / case_number * 100, correct_case, 100 * correct_questions/len(predict_results.keys())))
        print('The close acc is {:.6f}%, '
              'the open acc is {:.6f}%.'.
              format(correct_close / 416 * 100, correct_open / 645 * 100))

    print('Number of no test question image: ', no_test_q)
    print('Total image: ', case_number)

    if dataset_name == "RAD":
        return correct_case / case_number * 100, 100 * correct_questions / len(test_dataset.entries_test)
    else:
        return correct_case / case_number * 100, 100 * correct_questions / len(test_dataset.entries)


if __name__ == '__main__':
    args = parse_args()

    test_dataset = SLAKEGroupFeatureDataset('test', args, dataroot='./data-SLAKE')
    # test_dataset = RADGroupFeatureDataset('test', args, dataroot='./data-RAD')

    result_path = os.path.join("./saved_models/flitered_results_ereal.txt")

    f = open(result_path, "r", encoding='utf-8')
    lines = f.readlines()

    predict_results = {}
    for line in lines:
        qid = line.split(',tensor')[0].replace("tensor([", "").replace(".], device='cuda:0')", "")
        label = line.split(',tensor')[1].replace(", device='cuda:0')", "").replace("(", "").replace(")", "")

        predict_results[qid] = str(label)

    CaseAccuracy(predict_results, test_dataset, "SLAKE")