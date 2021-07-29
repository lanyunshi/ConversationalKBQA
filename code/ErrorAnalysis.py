import re
import os
import numpy as np
import random
import json
import argparse
from collections import defaultdict

def evaluation(data_path, file):
    a = []
    folder = os.path.basename(data_path)
    with open('data/test_%s/a.txt'%folder) as f:
        for line_idx, line in enumerate(f):
            a += [line.strip().lower().split('\t')]

    accuracies, precisions, recalls, F1s, hit1s = [], [], [], [], []
    with open(os.path.join(data_path, '%s_predcp.txt' %file)) as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if len(line.split('\t')) > 3:
                idx, g, F = line.split('\t')[:3]
                F = float(F[1: -1])
                ans = set(line.split('\t')[3:])
            else:
                ans = set([])
            acc, precision, recall, F1, hit1 = generate_evaluation_tmp(ans, set(a[line_idx]))
            #print(ans, a[line_idx], F1)
            #if line_idx == 77: print(F1); exit()
            accuracies += [acc]
            precisions += [precision]
            recalls += [recall]
            F1s += [F1]
            hit1s += [hit1]

    return np.mean(hit1s), np.mean(accuracies), np.mean(precisions), np.mean(recalls), np.mean(F1s)

def loop_evaluation(data_path, file):
    max_hit1, max_accuracy, max_precision, max_recall, max_F1 = 0, 0, 0, 0, 0
    for _ in range(1):
        hit1s, accuracies, precisions, recalls, F1s = evaluation(data_path, file)
        if hit1s > max_hit1:
            max_hit1, max_accuracy, max_precision, max_recall, max_F1 = hit1s, accuracies, precisions, recalls, F1s
    print('Hit@1: %s\nAccuracy: %s\nPrecision: %s\nRecall: %s\nF1s: %s' %(max_hit1, max_accuracy, max_precision, max_recall, max_F1))


def analyze_CSQA_boundary(data_path, ans):
    Q_file = json.load(open(os.path.join(data_path, 'SimpleRanker+all_history.json'), 'rb'))

    for line_idx, line in enumerate(Q_file):
        #if line_idx == limit_num: break
        print(str(line).encode('utf-8')); exit()
        #questions = line['questions']


def generate_hits1_result_from_json(data_path, file):
    a = []
    folder = os.path.basename(data_path)
    with open('data/test_%s/a.txt'%folder) as f:
        for line_idx, line in enumerate(f):
            a += [line.strip().lower().split('\t')]

    hits1, total_ans = [], []
    with open(os.path.join(data_path, '%s_predcp.json' %file)) as f:
        for line_idx, line in enumerate(f):
            line = json.loads(line)
            ans = defaultdict(int)

            new_line = {}
            for p in line:
                for an in line[p]:
                    new_line[an] = line[p][an]
            line = new_line

            one_ans = ''
            if len(line) == 0:
                hits1 += [0]
            else:
                last_idx = np.max([int(an[0]) for an in line])
                for an in line:
                    an_tmp = re.sub('^[012]\d', '', an)
                    if re.search('^[012]\d', an):
                        ans[an_tmp] += line[an]
                #print(ans)
                score = sorted(ans.values())[::-1][0]
                ans = set([an for an in ans if ans[an] == score])

                one_ans = random.sample(ans, 1)[0]
                hit1 = int(one_ans in a[line_idx])
                hits1 += [hit1]

            total_ans += [one_ans]
    #print(hits1)
    print('Hit@1: %s' %np.mean(hits1))
    return total_ans

def breakdown_evaluation(data_path, file):
    idx2category = []
    f = open('CONVEX/data/test_set/processed_q.json', 'r')
    lines = json.load(f)
    for line_idx, line in enumerate(lines):
        for q_idx, q in enumerate(line['questions']):
            idx2category += [line['domain']]
    #print('idx2category', idx2category); exit()
    category2acc, convidx2acc, category2upper, convidx2upper = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    total_acc, total_f1, te_acc, line2acc = [], [], [], defaultdict(list)
    error_idx = []

    preds = []
    with open(os.path.join(data_path, file + '_predcp.txt'), 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            #print(line.strip().split('\t')); exit()
            result = line.strip().split('\t')
            #print(line, result); exit()
            preds += [result]
    final_idx = result[0]
    final_idx, _ = final_idx.split('|')

    line_id = 0
    for line_idx in range(int(final_idx)):
        for time in range(5):
            result = preds[line_id]
            pred_rel, _, gold_rel = result[2: 5]
            pred_rel = pred_rel
            gold_rel = gold_rel
            #print(result, pred_rel, eval(pred_rel), [w for w in eval(pred_rel)]); exit()
            pred_te = [w for w in eval(pred_rel) if re.search('^Q\d', w)]
            gold_te = [w for w in eval(gold_rel) if re.search('^Q\d', w)]
            #print(pred_rel, gold_rel, pred_te, gold_te); exit()
            pred_idx, pred_time = preds[line_id][0].split('|')
            if str(line_idx) == str(pred_idx) and str(time) == str(pred_time):
                te_ac = int(set(pred_te) == set(gold_te))
                #print('result', result); exit()
                acc, upper = eval(result[-2])[0], float(eval(result[-1])[0])
                line_id += 1
            else:
                te_ac, acc, upper = 0, 0, 0
            te_acc += [te_ac]
            total_acc += [int(acc == 1)]
            total_f1 += [acc]
            category2acc[idx2category[line_idx]] += [int(acc == 1)]
            convidx2acc[str(time)] += [int(acc == 1)]
            category2upper[idx2category[line_idx]] += [int(upper == 1)]
            convidx2upper[str(time)] += [int(upper == 1)]
            line2acc[line_idx] += [int(acc == 1)]
            error_idx += [result[0]+' '+str(int(acc))]
    print('total acc', len(total_acc), np.mean(total_acc), 'total f1', np.mean(total_f1), 'te_acc', np.mean(te_acc), 'num', len(te_acc))
    sample_error = random.sample(error_idx, 100)
    # print(sample_error, np.sum([float(e.split()[-1]) for e in sample_error]))
    for category in ['movies', 'tv_series', 'music', 'books', 'soccer']:
        print(category, len(category2acc[category]), np.mean(category2acc[category]), np.mean(category2upper[category]))
    for convidx in convidx2acc:
        print(convidx, np.mean(convidx2acc[convidx]), np.mean(convidx2upper[convidx]))

def breakdown_csqa_evaluation(data_path, file):
    idx2category = {}
    f = open('CSQA/data/test_set/sequential_q.json', 'rb')
    lines = json.load(f)
    line_id = 1
    for line_idx, line in enumerate(lines):
        for q_idx, q in enumerate(line['questions']):
            #print('line', line)
            idx2category[(line_id, q_idx)] = q['ques_type_id']
        line_id += 1
    #print('idx2category', idx2category); exit()
    total_num = len(idx2category)
    category2acc, convidx2acc, category2upper, convidx2upper = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    category2f1 = defaultdict(list)
    total_acc, total_f1, total_upper = [], [], []
    prev_idx, line_id = 1, 0
    with open(os.path.join(data_path, file + '_predcp.txt'), encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            #print(line.strip().split('\t')); exit()
            result = line.strip().split('\t')

            if 'convcsqa' not in data_path and 'test' in file:
                idx = int(result[0])
                if idx != prev_idx:
                    line_id = 0
            else:
                idx, line_id = result[0].split('|')
                idx, line_id = int(idx), int(line_id)

            #print(idx)
            acc, upper = result[-2], float(eval(result[-1])[0])
            #print(eval(acc)); exit()
            total_acc += [int(eval(acc)[0] == 1)]
            total_f1 += [eval(acc)[0]]
            total_upper += [upper]
            category2acc[idx2category[(idx, line_id)]] += [int(eval(acc)[0] == 1)]
            category2f1[idx2category[(idx, line_id)]] += [eval(acc)[0]]
            convidx2acc[str(line_id)] += [int(eval(acc)[0] == 1)]
            category2upper[idx2category[(idx, line_id)]] += [int(upper == 1)]
            convidx2upper[str(line_id)] += [int(upper == 1)]

            prev_idx = idx
            line_id += 1
    print('total acc', len(total_acc), np.mean(total_acc), 'total f1', np.mean(total_f1), 'total_upper', np.mean(total_upper))
    for category in ['Simple Question (Direct)', 'Simple Question (Coreferenced)', 'Simple Question (Ellipsis)', 'Verification (Boolean) (All)']:
        print(category, len(category2acc[category]), np.mean(category2acc[category]), np.mean(category2f1[category]), np.mean(category2upper[category]))
    for convidx in convidx2acc:
        print(convidx, len(convidx2acc[convidx]), np.mean(convidx2acc[convidx]), np.mean(convidx2upper[convidx]))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_path", default=None, type=str, help="Path to data")
    parser.add_argument("--data_file", default=None, type=str, help="File of pred_cp data")
    parser.add_argument("--mode", default='eval', type=str, help="eval: evaluate [hit@1|accuracy|precision|recall|f1]; trans: transform the pred_cp file to official evaluation file")
    args = parser.parse_args()

    data_path = args.data_path
    data_file = args.data_file
    print(data_path, data_file)

    if 'CONVEX' in args.data_path:
        breakdown_csqa_convex_evaluation(data_path, data_file)
    elif 'csqa' in args.data_path:
        if args.mode == 'eval':
            loop_evaluation(data_path, data_file)
        elif args.mode == 'breakdown':
            breakdown_csqa_evaluation(data_path, data_file)
    elif 'convex' in args.data_path:
        if args.mode == 'eval':
            loop_evaluation(data_path, data_file)
        elif args.mode == 'breakdown':
            if 'CONVEX' in args.data_path:
                pass #breakdown_csqa_convex_evaluation(data_path, data_file)
            else:
                breakdown_evaluation(data_path, data_file)

'''

python3 code/ErrorAnalysis.py \
    --data_path trained_model/convex \
    --data_file RecurrentRanker+test+reproduce \
    --mode breakdown  \
    
'''
