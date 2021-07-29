# coding=utf-8
import logging
import numpy as np
import time as mytime
from datetime import datetime
from collections import defaultdict
import os
import re
import argparse
import torch
import random
import json
import copy
from torch import optim
import torch.nn as nn
from tqdm import tqdm, trange
from torch.distributions import Categorical
import torch.nn.functional as F

from tokenization import Tokenizer, BasicTokenizer
from library.ModelsRL import ModelConfig, Policy

from SPARQL_test import sparql_test
from dataset import *

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
const_minimax_dic = 'amount|number|how many|final|first|last|predominant|biggest|major|warmest|tallest|current|largest|most|newly|son|daughter' #
const_interaction_dic = '(and|or)'
const_verification_dic = '(do|is|does|are|did|was|were)'
my_sparql = sparql_test()


def Load_KB_Files(KB_file):
    """Load knowledge base related files.
    KB_file: {ent: {rel: {ent}}}; M2N_file: {mid: name}; QUERY_file: set(queries)
    """
    KB = json.load(open(KB_file, "r"))
    return KB

def Save_KB_Files(KB, KB_file):
    """Save knowledge base related files."""
    g = open(KB_file, "w")
    json.dump(KB, g)
    g.close()


def clean_answer(raw_answer, do_month =False):
    a = list(raw_answer)[0]
    try:
        if re.search('T', a) and do_month:
            #print(a)
            return list(set([datetime.strptime(a, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d") for a in raw_answer])), False
        elif re.search('T', a):
            return list(set([datetime.strptime(a, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y") for a in raw_answer])), False
        elif re.search('^\d+ \w+ \d{4}$', a):
            return list(set([datetime.strptime(a, "%d %B %Y").strftime("%Y-%m-%d") for a in raw_answer])), True
        elif re.search('^\d{4}$', a):
            return list(set([datetime.strptime(a, "%Y").strftime("%Y") for a in raw_answer])), False
    except:
        pass
    return list(raw_answer), False

def addin_historical_frontier(batch, first_topic_entity, previous_topic_frontier, previous_ans_frontier, tokenizer):
    '''
    Construct Entity Transition Graph
    :param batch: batch data
    :param first_topic_entity: the topic entity of the first turn of questions
    :param previous_topic_frontier: the topic entity of last turn of questions
    :param previous_ans_frontier: the answer entity of last turn of questions
    :param tokenizer: tokenizer
    :return: entity transition graph of this turn
    '''
    node_num = len(batch.historical_frontier)+len(first_topic_entity+previous_topic_frontier+previous_ans_frontier)

    batch.node_attri = [tokenizer.vocab['[unused5]']]*node_num # [unused5] --> other topic entity
    batch.out_degree_attri = [0]*node_num
    batch.type_attri = [927]*node_num
    batch.current_frontier = []
    for te in (first_topic_entity + previous_topic_frontier + previous_ans_frontier):
        if te not in batch.historical_frontier:
            batch.frontier2idx[te] = len(batch.historical_frontier)
            batch.historical_frontier += [te]
            batch.current_frontier += [te]
            batch.adjacent_matrix += [(batch.frontier2idx[te], batch.frontier2idx[te], 0)] # self-loop
            if te in previous_ans_frontier:
                #print('yeah')
                batch.adjacent_matrix += [(0, batch.frontier2idx[te], 1)] # previous answer entity links forward to first topic entity
                batch.adjacent_matrix += [(batch.frontier2idx[te], 0, 2)] # first topic entity links backward to previous answer entity
                for i in range(len(batch.historical_frontier)-2, 0, -1):
                    if batch.node_attri[i] != tokenizer.vocab['[unused7]']:
                        break
                    else:
                        batch.adjacent_matrix += [(i, batch.frontier2idx[te], 1)] # newly involved entity links forward to historical entity
                        batch.adjacent_matrix += [(batch.frontier2idx[te], i, 2)] # historical entity links backward to newly involved entity

        if te in first_topic_entity:
            batch.node_attri[batch.frontier2idx[te]] = tokenizer.vocab['[unused6]'] # [unused6] --> first topic entity
        elif te in previous_topic_frontier:
            batch.node_attri[batch.frontier2idx[te]] = tokenizer.vocab['[unused7]'] # [unused7] --> previous topic entity
        elif te in previous_ans_frontier:
            batch.node_attri[batch.frontier2idx[te]] = tokenizer.vocab['[unused8]'] # [unused8] --> previous answer entity

        batch.out_degree_attri[batch.frontier2idx[te]] = my_sparql.wikidata_id_to_out_degree(te) # out-degree attributes
        batch.type_attri[batch.frontier2idx[te]] = tokenizer.convert_tokens_to_ids([my_sparql.wikidata_id_to_type(te)])[0]
    batch.node_attri = batch.node_attri[:len(batch.historical_frontier)]
    batch.out_degree_attri = batch.out_degree_attri[:len(batch.historical_frontier)]
    batch.type_attri = batch.type_attri[:len(batch.historical_frontier)]
    # print('batch.frontier2idx', batch.frontier2idx, 'batch.historical_frontier', batch.historical_frontier)
    # print('first_topic_entity', first_topic_entity)
    # print('previous_topic_frontier', previous_topic_frontier)
    # print('previous_ans_frontier', previous_ans_frontier)
    # print('batch.historical_frontier', batch.historical_frontier)
    # print('batch.node_attri', batch.node_attri)
    # print('batch.adjacent_matrix', batch.adjacent_matrix)
    # print('batch.current_frontier', batch.current_frontier)

def retrieve_via_frontier(frontier, topic_entity, raw_candidate_paths, question=None, do_debug=False, not_update=True):
    '''
    Retrieve candidate relation paths based on entities
    :param frontier: entities in the entity transition graph
    :param raw_candidate_paths: a collection of candidate relation paths
    :param question: question of this round
    :param do_debug: whether it is debug mode
    :param not_update: whether update cache or not
    :return: candidate relation paths for the current round of questions
    '''
    if do_debug: print('frontier ****', frontier)

    if len(topic_entity) == 2 and re.search(const_interaction_dic, question):
        topic_entity = tuple(sorted(topic_entity))
        const_type = tuple(re.findall('(?<= )%s(?= )' %const_interaction_dic, question))

        key = (topic_entity, const_type)
        key = ' '.join([' '.join(list(r)) if isinstance(r, tuple) else str(r) for r in key])
        if const_type and (key not in my_sparql.STATEMENTS):
            if not_update:
                statements = {}
            else:
                statements, sparql_txts = my_sparql.SQL_1hop_interaction(((topic_entity[0],), (topic_entity[1],)), const_type)
                my_sparql.QUERY_TXT = my_sparql.QUERY_TXT.union(sparql_txts)
                my_sparql.STATEMENTS[key].update(statements)
            #print('my_sparql.STATEMENTS[(t, const_type)], mid', key, mytime.time()-time1)
        else:
            statements = my_sparql.STATEMENTS[key]
            # print('cache my_sparql.STATEMENTS[(t, const_type)], mid', mytime.time()-time1)
        if statements: raw_candidate_paths += [statements]
        #print('raw_candidate_paths', raw_candidate_paths); exit()

    for t in set(frontier):
        if not re.search('^Q', t): continue
        #print('my_sparql.STATEMENTS[(t, None)]')
        time1 = mytime.time()
        key = (t, None)
        key = ' '.join([' '.join(list(r)) if isinstance(r, tuple) else str(r) for r in key])

        if key not in my_sparql.STATEMENTS:
            if not_update:
                statements = {}
            else:
                # print('t', t)
                statements, sparql_txts = my_sparql.SQL_1hop(((t,),), my_sparql.QUERY_TXT)
                #print('statements, sparql_txts', statements, sparql_txts)
                my_sparql.QUERY_TXT = my_sparql.QUERY_TXT.union(sparql_txts)
                statements_tmp, sparql_txts = my_sparql.SQL_2hop(((t,),), my_sparql.QUERY_TXT)
                my_sparql.QUERY_TXT = my_sparql.QUERY_TXT.union(sparql_txts)
                statements.update(statements_tmp)
                my_sparql.STATEMENTS[key].update(statements)
                #print('my_sparql.STATEMENTS[(t, None)]', key, mytime.time()-time1, statements)
        else:
            statements = my_sparql.STATEMENTS[key]
            #print('cache my_sparql.STATEMENTS[(t, None)]', mytime.time()-time1, statements)
        if statements: raw_candidate_paths += [statements]

        # If multiple entities involve in a question, other entities are treated as the constraints
        sorted_topic_entity = tuple(sorted(set(frontier) - set([t])))
        if len(sorted_topic_entity):
            time1 = mytime.time()
            key = (t, sorted_topic_entity)
            key = ' '.join([' '.join(list(r)) if isinstance(r, tuple) else str(r) for r in key])
            if key not in my_sparql.STATEMENTS:
                if not_update:
                    statements = {}
                else:
                    statements, sparql_txts = my_sparql.SQL_2hop_reverse(((t,),), set(frontier) - set([t]))
                    my_sparql.QUERY_TXT = my_sparql.QUERY_TXT.union(sparql_txts)
                    my_sparql.STATEMENTS[key].update(statements)
                #print('my_sparql.STATEMENTS[(t, sorted_topic_entity)]', mytime.time()-time1)
            else:
                statements = my_sparql.STATEMENTS[key]
                #print('cache my_sparql.STATEMENTS[(t, sorted_topic_entity)]', mytime.time()-time1)
            if statements: raw_candidate_paths += [statements]

        # If min max keyword in the question, we treat it as the constraint
        if question is not None and re.search(const_minimax_dic, question):
            const_type = tuple(re.findall(const_minimax_dic, question))
            time1 = mytime.time()
            key = (t, const_type)
            key = ' '.join([' '.join(list(r)) if isinstance(r, tuple) else str(r) for r in key])
            if key not in my_sparql.STATEMENTS:
                if not_update:
                    statements = {}
                else:
                    statements, sparql_txts = my_sparql.SQL_1hop_reverse(((t,),), const_type)
                    my_sparql.QUERY_TXT = my_sparql.QUERY_TXT.union(sparql_txts)
                    my_sparql.STATEMENTS[key].update(statements)
                #print('my_sparql.STATEMENTS[(t, const_type)], mid', key, mytime.time()-time1)
            else:
                statements = my_sparql.STATEMENTS[key]
                #print('cache my_sparql.STATEMENTS[(t, const_type)], mid', mytime.time()-time1)
            if statements: raw_candidate_paths += [statements]

        #print('my_sparql.STATEMENTS[(t, const_type)], year')
        # If the year string in the question, we should treat it as constraint
        if question is not None and re.search('[0-9][0-9][0-9][0-9]', question):
            const_type = tuple(re.findall('[0-9][0-9][0-9][0-9]', question))
            time1 = mytime.time()
            if (t, const_type) not in my_sparql.STATEMENTS:
                if not_update:
                    statements = {}
                else:
                    statements, sparql_txts = my_sparql.SQL_2hop_reverse(((t,),), const_type)
                    my_sparql.QUERY_TXT = my_sparql.QUERY_TXT.union(sparql_txts)
                    my_sparql.STATEMENTS[(t, const_type)].update(statements)
                #print('my_sparql.STATEMENTS[(t, const_type)], year', mytime.time()-time1)
            else:
                statements = my_sparql.STATEMENTS[(t, const_type)]
                #print('cache my_sparql.STATEMENTS[(t, const_type)], year', mytime.time()-time1)
            if statements: raw_candidate_paths += [statements]
    return raw_candidate_paths


def retrieve_KB(batch, tokenizer, config, do_debug=False, time = 0, is_train=True, not_update=False):
    '''
    Retrieve subgraphs from the KB based on current topic entities
    :param batch: batch data
    :param tokenizer: tokenizer
    :param config: method configuration
    :return: retrieve all candidate relation paths of this turn
    '''
    raw_candidate_paths, paths, batch.orig_F1s, topic_entities =[], {}, [], []
    time1 = mytime.time()

    if time == 0:
        # retrieve from the topic entity of the first round
        topic_entity = batch.topic_entity
        raw_candidate_paths = retrieve_via_frontier(topic_entity, topic_entity,
                                                    raw_candidate_paths, do_debug=do_debug,
                                                    not_update=not_update)
        batch.frontier2typeidx = frontier2typeidx = [[], [], [], batch.topic_entity]
        batch.current_frontier = topic_entity
    else:
        # retrieve from topic entity of current round
        topic_entity = batch.question[time]['ner']

        # retrieve from answer entity of previous round
        ans_limit = 1 if tokenizer.dataset == 'CONVEX' else 3
        ans_frontier = sum([list(batch.path2ans[t])[:ans_limit] for t in batch.path2ans], []) #[:1]
        ans_frontier = [t for t in ans_frontier if t if re.search('^Q', t)]

        # retrieve from topic entity of previous round
        previous_topic_frontier = sum([[w for w in t if re.search('^Q', w)] for t in batch.path2ans], [])
        sorted_topic_entity = tuple(sorted(list(set(previous_topic_frontier + batch.topic_entity))))

        if not is_train: #tokenizer.dataset == 'CSQA':
            topic_entity = [t if re.search('^Q\d', t) else my_sparql.wikidata_label_to_id(t) for t in topic_entity]
        else:
            topic_entity = [t if re.search('^Q\d', t) else my_sparql.wikidata_label_to_id(t, sorted_topic_entity) for t in topic_entity]

        topic_entity = [t for t in topic_entity if t not in [u'UNK']]
        batch.current_topics = topic_entity

        # construct entity transition graph
        addin_historical_frontier(batch, batch.topic_entity, previous_topic_frontier, ans_frontier, tokenizer)
        if previous_topic_frontier == batch.topic_entity:
            previous_topic_frontier_tmp = []
        else:
            previous_topic_frontier_tmp = copy.deepcopy(previous_topic_frontier)

        # it is verification question
        if re.search('^%s' % const_verification_dic, batch.question[time]['question']) and len(set(topic_entity)) > 0:
            frontier = set(topic_entity)
        else:
            frontier = set(topic_entity + batch.historical_frontier)

        # newly added entites in the entity transition graph
        other_entity = list(set(batch.historical_frontier) - set(previous_topic_frontier_tmp + ans_frontier + batch.topic_entity))
        batch.frontier2typeidx = frontier2typeidx = [previous_topic_frontier_tmp, ans_frontier, other_entity, batch.topic_entity]

        raw_candidate_paths = retrieve_via_frontier(frontier, topic_entity,
                                                    raw_candidate_paths, batch.question[time]['question'],
                                                    do_debug=do_debug, not_update=not_update)


    # process the candidate paths
    candidate_paths, topic_scores, topic_numbers, answer_numbers, type_numbers, superlative_numbers, year_numbers, hop_numbers, F1s, RAs = [], [], [], [], [], [], [], [], [], []
    max_cp_length, types, path2ans = 0, [], {}
    limit_number = 1000

    for s_idx, statements in enumerate(raw_candidate_paths):
        filter_statements = {}
        for p_idx, p in enumerate(statements):
            if len(statements[p]) > 3 and tokenizer.dataset == 'CONVEX': continue
            path2ans[sum(p, ())] = (statements[p], 0)
            filter_statements[p] = statements[p]
        batch.statements += [filter_statements]

    batch.path2ans = path2ans
    sorted_path = sorted(batch.path2ans.keys())

    # If it is verification question, we start from the historical topic entity and see whether current topic entity is retrieved
    gold_ans, do_month = clean_answer(batch.question[time]['answer'])
    if re.search('^%s' %const_verification_dic, batch.question[time]['question']):
        gold_ans = [w for w in batch.question[time]['relation']] if batch.question[time]['relation'] != '' else batch.current_topics
        pseudo_ans = [w.lower() for w in batch.question[time]['answer_text']]

    batch.topic_F1s = [0] * len(batch.frontier2typeidx)
    if config.use_te_graph:
        batch.topic_F1s = [0] * len(batch.historical_frontier)
        batch.current_frontier_idx = []
        for raw_te in batch.current_frontier:
            if not re.search('^\?', raw_te): te = my_sparql.wikidata_id_to_label(raw_te)

            te = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(te))
            batch.current_frontier_idx += [te]

    # extract the features of candidate paths
    for p_idx, p in enumerate(sorted_path):
        pred_ans, _ = clean_answer(path2ans[p][0], do_month)
        if re.search('^%s' % const_verification_dic, batch.question[time]['question']) and is_train:
            measure_F1 = generate_Inclusion(gold_ans, set(p))
        elif re.search('^%s' % const_verification_dic, batch.question[time]['question']):
            measure_F1 = ["yes"] if generate_Inclusion(gold_ans, set(p)) == 1 else ["no"]
            if len(set(topic_entity)) == 0 : measure_F1 = ["yes"]
            measure_F1 = np.float(measure_F1 == pseudo_ans)
        else:
            measure_F1 = generate_F1(pred_ans, gold_ans)
        p_txt = [my_sparql.wikidata_id_to_label(w) for w in p if not re.search('^\?', w)]

        if (not is_train) or (np.random.random() < (limit_number*1./len(path2ans))) or measure_F1 > 0.5:
            if None in p_txt: continue

            path, topic_score, p_tmp, topic_number, type_number, superlative_number, year_number, contain_minimax, skip = [], [], (), 0., 0., 0., 0., False, False
            path = tokenizer.tokenize(' '.join(p_txt))

            if do_debug and measure_F1: print('path', time, p, str(p_txt).encode('utf8'), pred_ans[:3], measure_F1) # I comment this out!
            path = tokenizer.convert_tokens_to_ids(path)
            try:
                pred_ans = tokenizer.tokenize(' '.join(pred_ans[:1]).decode())[:5]
                pred_ans = tokenizer.convert_tokens_to_ids(pred_ans)
            except:
                pred_ans = [100]
            answer = np.log(len(path2ans[p]))

            # Append features for ranking
            candidate_paths += [path]  # text of candidate path
            batch.candidate_paths += [p_txt]
            batch.orig_candidate_paths += [p]
            topic_scores += [topic_score] # accumulative scores of topic entity in the candidate path
            hop_numbers += [pred_ans]

            F1 = measure_F1
            batch.current_F1s += [F1]
            batch.orig_F1s += [F1]
            batch.F1s += [F1]

            if len(path) > max_cp_length:
                max_cp_length = len(path)

    # normalize the supervision array of answers for training
    batch.current_F1s, batch.F1s = np.array(batch.current_F1s), np.array(batch.F1s)
    if np.sum(batch.current_F1s) == 0: batch.current_F1s[:] = 1.
    if np.sum(batch.F1s) == 0: batch.F1s[:] = 1.
    batch.current_F1s /= np.sum(batch.current_F1s)
    batch.F1s /= np.sum(batch.F1s)

    historical_frontier = batch.historical_frontier if config.use_te_graph else batch.frontier2typeidx
    filtered_historical_frontier = sum(batch.frontier2typeidx, []) if config.use_te_graph else batch.historical_frontier

    # Find the psuedo gold topic entities where the candidate path gives maximum F1 score
    if len(batch.F1s):
        max_F1 = np.max(batch.orig_F1s)
        max_idx_entities = sum([batch.orig_candidate_paths[idx] for idx, F1 in enumerate(batch.orig_F1s) if F1 == max_F1], ())
        if max_F1 > 0:
            for f_idx, frontier in enumerate(historical_frontier):
                if not isinstance(frontier, list): frontier = [frontier]
                if len(set(max_idx_entities) & set(frontier)):
                    max_frontier = (set(max_idx_entities) & set(frontier))
                    if list(max_frontier)[0] in filtered_historical_frontier:
                        if f_idx == 0:
                            batch.topic_F1s[f_idx] = 1;
                        else:
                            batch.topic_F1s[f_idx] = 5

        # revisit frontier: map the topic indexes of each relation path
        for t_idx, _ in enumerate(topic_scores):
            for f_idx, frontier in enumerate(historical_frontier):
                if not isinstance(frontier, list): frontier = [frontier]
                if len(set(batch.orig_candidate_paths[t_idx]) & set(frontier)):
                    topic_scores[t_idx] += [f_idx];
            if len(set(batch.orig_candidate_paths[t_idx]) & set(batch.current_topics)):
                topic_scores[t_idx] += [-1]

    return candidate_paths, topic_scores, hop_numbers, max_cp_length

def select_te_field(batch, device):
    if len(batch.current_frontier_idx):
        f = np.max([len(f) for f in batch.current_frontier_idx])
        tn = torch.tensor(truncated_sequence(batch.current_frontier_idx, f), dtype=torch.long).to(device)
    else:
        tn = []
    m = len(batch.node_attri)
    adjacent_matrix = np.zeros((m, m, 3)) # self-loop, forward and backward links
    for idx in batch.adjacent_matrix:
        i, j, k = idx
        adjacent_matrix[i, j, k] = 1

    ty_n = torch.tensor(adjacent_matrix, dtype=torch.float).view(m, m, -1).to(device)
    su_n = torch.tensor(batch.node_attri, dtype=torch.long).view(-1).to(device)
    out_n = torch.tensor(batch.out_degree_attri, dtype=torch.float).view(-1).to(device)
    type_n = torch.tensor(batch.type_attri, dtype=torch.long).view(-1).to(device)
    return (tn, ty_n, su_n, out_n, type_n)

def select_field(q, cp, ts, hn, mcl, batch, config, ts_top=None):
    # candidate_paths, topic_scores, 0, 0, 0, 0, 0, hop_numbers, max_cp_length
    # if we do filtering to the topic entity, we should activate the candidate path with these filtered topic entity
    if ts_top is not None:
        cp = [x for x_idx, x in enumerate(cp) if x_idx in ts_top]
        ts = [x for x_idx, x in enumerate(ts) if x_idx in ts_top]
        hn = [x for x_idx, x in enumerate(hn) if x_idx in ts_top]
        batch.F1s = [x for x_idx, x in enumerate(batch.F1s) if x_idx in ts_top]
        batch.candidate_paths = [x for x_idx, x in enumerate(batch.candidate_paths) if x_idx in ts_top]
        batch.orig_candidate_paths = [x for x_idx, x in enumerate(batch.orig_candidate_paths) if x_idx in ts_top]
        batch.current_F1s = [x for x_idx, x in enumerate(batch.current_F1s) if x_idx in ts_top]
        batch.orig_F1s = [x for x_idx, x in enumerate(batch.orig_F1s) if x_idx in ts_top]

    if config.method in ['RecurrentRanker', 'SimpleRecurrentRanker']:
        mcl = np.min([np.max([mcl, len(q), np.max([len(h) for h in hn])]), 50])

        sequence, sequence_token, answer = [], [], []
        for i in range(len(cp)):
            sequence += [[101] + q] if config.use_bert else [q]
            sequence_token += [[101] + cp[i]] if config.use_bert else [cp[i]]
            answer += [[101] + hn[i]] if config.use_bert else [hn[i]]

        cp = truncated_sequence(sequence_token, mcl)
        q = truncated_sequence(sequence, mcl)
        hn = truncated_sequence(answer, mcl)
    else:
        mcl = np.min([np.max([mcl, len(q)]), 50])

        sequence, sequence_token = [], []
        for i in range(len(cp)):
            sequence += [[101] + q]
            sequence_token += [[101] + cp[i]]

        cp = truncated_sequence(sequence_token, mcl)
        q = truncated_sequence(sequence, mcl)
        hn = truncated_sequence(sequence_token, mcl)

    q = torch.tensor(q, dtype=torch.long).view(1, len(q), -1)
    cp = torch.tensor(cp, dtype=torch.long).view(1, len(cp), -1)#

    if config.method in ['RecurrentRanker', 'SimpleRecurrentRanker']:
        hn = torch.tensor(hn, dtype=torch.long).view(1, len(hn), -1)
        ts = torch.tensor(ts, dtype=torch.float).view(1, -1)
    else:
        hn = torch.tensor(hn, dtype=torch.long).view(1, len(hn), -1)
        ts = torch.tensor(ts, dtype=torch.float).view(1, -1)
    return q, cp, ts, hn

def edit_ts(ts, te_logits, time=0):
    new_ts = [0] * len(ts)
    if time > 0 and te_logits is not None:
        for t_idx, _ in enumerate(ts):
            new_ts[t_idx] = np.sum([1 if idx==-1 else te_logits[idx] for idx in ts[t_idx]])
    return new_ts

def filter_ts(ts, te_logits, time=0):
    ts_top = []
    sum_ts = set(sum(ts, []))
    if len(sum_ts) == 0: return list(np.arange(len(ts)))
    if te_logits is None:
        top_idx = random.sample(sum_ts, np.min([3, len(sum_ts)])) #np.random.rand(len(set(sum(ts, [])) - set([-1])))
    else:
        top_idx = list(np.argsort(te_logits[:len(sum_ts - set([-1]))])[::-1][:3])
    for t_idx, _ in enumerate(ts):
        if len(set(ts[t_idx]) & set(top_idx)) or len(set(ts[t_idx]) & set([-1])):
            ts_top += [t_idx]
    return ts_top

def truncated_sequence(cp, mcl, fill=0):
    for c_idx, c in enumerate(cp):
        if len(c) > mcl:
            cp[c_idx] = c[:mcl]
        elif len(c) < mcl:
            cp[c_idx] += [fill] * (mcl - len(c))
    return cp

def select_action(policy, raw_logits, adjust_F1s = None, is_train = True, time =0, k = 1, is_reinforce=True):
    #Select an action (0 or 1) by running policy model and choosing based on the probabilities in state
    loss = None
    if raw_logits is None:
        '''If we don't have positive examples'''
        action = torch.argsort(adjust_F1s, dim=1, descending=True)[:, :k].view(1, k)
        loss = 0
    elif is_train and is_reinforce == 2:
        '''If we use semi-supervision with beam'''
        logits = F.softmax(raw_logits, 1)
        if torch.isnan(logits).any(): print(logits[:10]); exit() # debug
        k = np.min([k, logits.size(1)])# if time <2 else np.min([100, logits.size(1)])
        loss = nn.KLDivLoss(reduction='sum')(logits.log(), adjust_F1s)
        action = torch.argsort(adjust_F1s, dim =1, descending=True)[:, :k].view(1, k)
    elif is_train:
        '''If we use reinforcement learning'''
        k = 3
        logits = F.softmax(raw_logits, 1)
        probs = 0.5*logits + 0.5*adjust_F1s  # #
        if torch.isnan(probs).any(): print(probs[:10]); exit() # debug
        c = Categorical(probs=probs)
        action = c.sample((k, ))
        #action = torch.argsort(raw_logits, dim =1, descending=True).view(1, -1)
        c_log_prob = torch.gather(probs.transpose(1, 0), 0, action)
        # Add log probability of our chosen action to our history
        if policy.policy_history.dim() != 0:
            policy.policy_history = torch.cat([policy.policy_history, c_log_prob.view(k, 1)], -1)
        else:
            policy.policy_history = c_log_prob.view(k, 1)
    else:
        k = np.min([k, raw_logits.size(1)]) #if time <2 else np.min([100, raw_logits.size(1)])
        logits = adjust_F1s if adjust_F1s is not None else raw_logits
        action = torch.argsort(logits, dim =1, descending=True)[:, :k].view(k, 1)
    return action, loss

def update_train_instance(batch):
    batch.candidate_paths = []
    batch.orig_candidate_paths = []
    batch.current_F1s = []
    batch.orig_F1s = []
    batch.F1s = []

def generate_reward(logits, action, batch, is_train = False, do_debug = False, eval_metric = 'AnsAcc', top_pred_ans = None):
    if top_pred_ans is None or len(top_pred_ans) == 0: top_pred_ans = defaultdict(int)

    action_F1s, F1, pred_cp, pred_ans, new_path2ans, new_statements, F1s = [], 0, '', '', {}, [], [0]
    for a in action.reshape(-1):
        if a < len(batch.candidate_paths):
            pred_cp = batch.orig_candidate_paths[a]
            pred_ans = batch.path2ans[pred_cp][0]
            new_path2ans[pred_cp] = pred_ans
            new_statements += [batch.statements[batch.path2ans[pred_cp][1]]]
            pred_cp = str(pred_cp)+'\t'+str(batch.candidate_paths[a])
            if eval_metric in ['F1', 'Hits1']:
                F1 = batch.orig_F1s[a]
            else:
                raise Exception('Evaluation metric is not correct !')
        action_F1s += [F1]
    batch.path2ans = new_path2ans
    batch.statements = []

    if is_train and (logits is not None):
        max_idx = np.argmax(logits)
        F1s = [batch.orig_F1s[max_idx]]
    else:
        max_idx = np.argmax(batch.orig_F1s)
        pred_cp = pred_cp + '\t' + str(batch.orig_candidate_paths[max_idx])+'\t'+str(batch.historical_frontier)+'\t'+str(list(pred_ans)[:3])
        F1s = [batch.orig_F1s[max_idx]]

    return action_F1s, pred_cp.encode('utf-8'), F1s, list(pred_ans)[:3]

def generate_Acc(pred_graph, golden_graph):
    Acc = float(set(pred_graph) == set(golden_graph))
    return Acc


def generate_Inclusion(pred_ans, gold_ans):
    return float(set(pred_ans).issubset(gold_ans))

def generate_F1(pred_ans, ans):
    TP = len(set(pred_ans) & set(ans))
    precision = TP*1./np.max([len(set(pred_ans)), 1e-10])
    recall = TP*1./np.max([len(set(ans)), 1e-10])
    F1 = 2. * precision * recall/np.max([(precision + recall), 1e-10])
    return F1

def update_policy_immediately(adjust_loss, optimizer, te_losses=0):
    # Update network weights
    optimizer.zero_grad()
    adjust_loss.backward(retain_graph=True) #
    optimizer.step()

    return adjust_loss.item()#, te_losses

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--train_folder", default=None, type=str, help="QA folder for training. E.g., train")
    parser.add_argument("--dev_folder", default=None, type=str, help="QA folder for dev. E.g., dev")
    parser.add_argument("--test_folder", default=None, type=str, help="QA folder for test. E.g., test")
    parser.add_argument("--vocab_file", default=None, type=str, help="Vocab txt for vocabulary")
    parser.add_argument("--cache_dir", default=None, type=str, help="Cache folder for loading")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints will be written")

    # Other parameters
    parser.add_argument("--load_model", default=None, type=str, help="The pre-trained model to load")
    parser.add_argument("--save_model", default='BaseSave', type=str, help="The name that the models save as")
    parser.add_argument("--config", default='config/base_config.json', help="The config of base model")
    parser.add_argument("--num_train_epochs", default=20, type=int, help="The epoches of training")
    parser.add_argument("--do_train", default=1, type=int, help="Whether to run training")
    parser.add_argument("--do_eval", default=1, type=int, help= "Whether to run eval")
    parser.add_argument("--train_batch_size", default=1, type=int, help="Total batch size for training")
    parser.add_argument("--eval_batch_size", default=1, type=int, help="Total batch size for eval")
    parser.add_argument("--learning_rate", default=5e-6, type=float, help="Total number of training epoches to perform")
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Proportion of training to perform linear learning rate warmup for.")
    parser.add_argument("--seed", default=123, type=int, help="random seeed for initialization")
    parser.add_argument("--gpu_id", default=1, type=int, help="id of gpu")
    parser.add_argument("--top_k", default=1, type=int, help="retrieve top k relation path during prediction")
    parser.add_argument("--do_debug", default=0, type=int, help="whether in debug mode")
    parser.add_argument("--do_policy_gradient", default=1, type=int, help="Whether to train with policy gradient. 1: use policy gradient; 2: use maximum likelihood with beam")
    args = parser.parse_args()

    if torch.cuda.is_available():
        logger.info("cuda {} is available".format(args.gpu_id))
        device = torch.device("cuda", args.gpu_id) #
        args.gpu_id = 1
    else:
        device = None
        logger.info("cuda is unavailable")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    load_model_file = args.load_model+".bin" if args.load_model else None
    save_model_file = os.path.join(args.output_dir, args.save_model+".bin") if args.save_model else os.path.join(args.output_dir, "base_model.bin")
    save_eval_cp_file = os.path.join(args.output_dir, args.save_model+"_predcp.txt")
    save_eval_file = os.path.join(args.output_dir, args.save_model+".log")

    config = ModelConfig.from_json_file(args.config)
    tokenizer = Tokenizer(args.vocab_file) if config.use_bert else BasicTokenizer(args.vocab_file)
    tokenizer.dataset = args.train_folder.split('/')[0]

    cache_dir = re.sub('\_v\d', '_debug', args.cache_dir) if args.do_debug else args.cache_dir
    my_sparql.load_cache('%s/M2N.json' % cache_dir,
                         '%s/STATEMENTS.json' % cache_dir,
                         '%s/QUERY.json' % cache_dir,
                         '%s/TYPE.json' % cache_dir,
                         '%s/OUTDEGREE.json' % cache_dir)

    policy = Policy(config, tokenizer.vocab, device)
    if load_model_file and os.path.exists(load_model_file):
        model_dic = torch.load(load_model_file, map_location='cpu')
        #model_dic = {k:torch.cat([model_dic[k][:, :1], model_dic[k][:, -1:]], 1) if k == 'ranker.decoder.weight' else model_dic[k]  for k in model_dic}
        policy.load_state_dict(model_dic, strict=True)
        print("successfully load pre-trained model ...");
    else:
        print("successfully initialize model ...")

    if args.gpu_id:
        policy.to(device)

    global_step, max_eval_reward, t_total = 0, -0.1, 0
    if args.do_eval:
        dev_instances = create_instances(input_file=args.dev_folder,
                                          tokenizer=tokenizer)
        test_instances = create_instances(input_file=args.test_folder,
                                          tokenizer=tokenizer)
        logger.info("***** Loading evaluation *****")
        logger.info("   Num dev examples = %d", len(dev_instances))
        logger.info("   Num test examples = %d", len(test_instances))
        logger.info("   Batch size = %s", args.eval_batch_size)
    if args.do_train:
        train_instances = create_instances(input_file=args.train_folder,
                                           tokenizer=tokenizer)
        logger.info("***** Loading training ******")
        logger.info("    Num examples = %d" , len(train_instances))
        logger.info("    Batch size = %s", args.train_batch_size)
        t_total = len(train_instances)*args.num_train_epochs

    # Prepare optimizer
    param_optimizer = list(policy.parameters())
    optimizer = optim.Adam(param_optimizer, lr=args.learning_rate)

    args.num_train_epochs = 1 if not args.do_train else args.num_train_epochs
    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):

        tr_loss, tr_te_boundary, tr_reward, tr_reward_boundary = 0., 0., 0., 0.,
        hop1_tr_reward, nb_tr_examples, nb_tr_steps, query_num = 0, 0, 0, 0.
        tr_te_loss, tr_te_reward = 0., 0.

        if args.do_train:
            policy.train()
            if args.do_debug: train_instances = train_instances[:1]

            # shuffle the training data
            random.shuffle(train_instances)

            for step in trange(0, np.min([500, len(train_instances)]), desc="Train"):
                batch = train_instances[step]
                time, _total_losses = 0, 0

                while time < len(batch.question):
                    update_train_instance(batch)

                    # Retrieve graphs based on the current graph
                    cp, ts, hn, mcl = retrieve_KB(batch, tokenizer, config, do_debug=args.do_debug,
                                            time = time, is_train = True,
                                            not_update = args.do_train)

                    te_loss, te_acc = 0, 0
                    q = batch.question[time]['question_idx']
                    te_logits = None

                    # construct entity transition graph
                    with torch.autograd.set_detect_anomaly(True):
                        if config.method in ['RecurrentRanker', 'SimpleRecurrentRanker'] and time > 0:
                            # update conversation history encoder
                            policy.ranker.forward_answer()

                            if config.method in ['RecurrentRanker']:
                                # train the focal entity predictor
                                te_batch = select_te_field(batch, device)
                                _te_loss, te_acc, max_te = policy.ranker.train_next_te(batch.topic_F1s, te_batch, time)
                                if time > 0 and config.method in ['RecurrentRanker']:
                                    te_logits = policy.ranker.te_logits

                                # Update the parameters in focal entity predictor
                                te_loss = update_policy_immediately(_te_loss, optimizer)

                        # Inject the focal entity information into the batch data
                        ts_top = None #filter_ts(ts, te_logits, time = time)
                        ts = edit_ts(ts, te_logits, time = time)

                        # When there is no candidate paths for the question, skip
                        if len(cp) == 0: break
                        if True: #np.sum(batch.orig_F1s):
                            ready_batch = select_field(q, cp, ts, hn, mcl, batch, config,
                                                       ts_top=ts_top)
                            if args.gpu_id: ready_batch = tuple(t.to(device) for t in ready_batch)

                            # Do single-turn KBQA
                            if config.method in ['RecurrentRanker', 'SimpleRecurrentRanker']:
                                if time == 0: policy.ranker.initial_hidden_state()
                            _logits, _losses = policy.ranker(ready_batch, time)

                            logits = _logits.cpu().data.numpy() if args.gpu_id else _logits.data.numpy()
                            F1s = torch.tensor(batch.F1s, dtype=torch.float).view(1, -1)
                            if args.gpu_id: _F1s = F1s.to(device)

                            # When there is a bug, skip
                            if torch.isnan(_logits).any(): break

                            _action, _adjust_loss = select_action(policy, _logits, adjust_F1s = _F1s,
                                                is_train=True, time = time, is_reinforce=args.do_policy_gradient)
                            if args.do_policy_gradient==2: loss = update_policy_immediately(_adjust_loss, optimizer)

                        # Select the top-ranked entity as the answer entity
                        if config.method in ['RecurrentRanker', 'SimpleRecurrentRanker']:
                            policy.ranker.update_answer(_action)

                        # Evaluate the answer
                        action = _action.cpu().data.numpy() if args.gpu_id else _action.data.numpy()
                        _, _, reward, _ = generate_reward(logits, action, batch,
                                                          is_train=True, do_debug=args.do_debug,
                                                          eval_metric='F1')
                        # Save reward
                        tr_loss += loss
                        if time > 0: tr_te_loss += te_loss
                        tr_te_reward += te_acc
                        tr_te_boundary += np.max(batch.topic_F1s) if len(batch.topic_F1s) else 0.
                        tr_reward_boundary += np.max(batch.orig_F1s)
                        tr_reward += np.mean(reward)
                        nb_tr_examples += 1
                        if time > 0: nb_tr_steps += 1
                        global_step += 1
                        time += 1
                        policy.reward_episode.append(reward)

                if args.do_debug: print('tr_loss', tr_loss, 'tr_reward_boundary', tr_reward_boundary); #exit()
                policy.reset()
                batch.reset()

        if args.do_eval:
            policy.eval()
            eval_reward, eval_reward_boundary, nb_eval_steps = 0, 0, 0
            nb_eval_examples, eval_te_reward, eval_te_boundary = 0, 0, 0
            
            if args.do_eval == 2: dev_instances = dev_instances[:1]
            if not args.do_train: dev_instances = dev_instances[:10]
            if args.do_debug: dev_instances = dev_instances[:0] #100

            for eval_step in trange(np.min([3000, len(dev_instances)]), desc="Dev"):  #len(dev_instances)
                batch = dev_instances[eval_step]
                time, time1, pred_cp = 0, mytime.time(), ''
                
                while time < len(batch.question):
                    update_train_instance(batch)

                    # Retrieve graphs based on the current graph
                    cp, ts, hn, mcl = retrieve_KB(batch, tokenizer, config, do_debug=args.do_debug,
                                                    time = time, is_train=False,
                                                    not_update = args.do_train)

                    q = batch.question[time]['question_idx']
                    te_logits, te_acc = None, 0
                    if config.method in ['RecurrentRanker', 'SimpleRecurrentRanker'] and time > 0:
                        
                        policy.ranker.forward_answer()
                        
                        if config.method in ['RecurrentRanker']:
                            te_batch = select_te_field(batch, device)
                            _, te_acc, max_te = policy.ranker.predict_next_te(batch.topic_F1s, te_batch, time)
                            if time > 0 and config.method in ['RecurrentRanker']: #, 'RecurrentRanker_match'
                                te_logits = policy.ranker.te_logits

                    ts_top = None #filter_ts(ts, te_logits, time = time)
                    ts = edit_ts(ts, te_logits, time = time) #te_logits batch.topic_F1s

                    if len(cp) == 0: break # When there is no candidate paths for the question, skip
                    ready_batch = select_field(q, cp, ts, hn, mcl, batch, config, ts_top=ts_top)
                    if args.gpu_id: ready_batch = tuple(t.to(device) for t in ready_batch)

                    # Step through environment using chosen action
                    with torch.no_grad():
                        if config.method in ['RecurrentRanker', 'SimpleRecurrentRanker']:
                            if time == 0: policy.ranker.initial_hidden_state()
                            _logits, _losses = policy.ranker(ready_batch, time)
                        else:
                            _logits, _ = policy.ranker(ready_batch, time)
                        _logits = F.softmax(_logits, 1)

                    logits = _logits.cpu().data.numpy() if args.gpu_id else _logits.data.numpy()
                    F1s = torch.tensor(batch.F1s, dtype=torch.float).view(1, -1).to(device)
                    if args.gpu_id: _F1s = F1s.to(device)

                    _action, _adjust_loss = select_action(policy, _logits, adjust_F1s = None,
                                         is_train=False, time = time, is_reinforce=args.do_policy_gradient)

                    if config.method in ['RecurrentRanker', 'SimpleRecurrentRanker']:
                        policy.ranker.update_answer(_action)

                    action = _action.cpu().data.numpy() if args.gpu_id else _action.data.numpy()
                    reward, _, _, _ = generate_reward(logits, action, batch, do_debug=args.do_debug, eval_metric='F1')

                    eval_reward += np.mean(reward)
                    eval_reward_boundary += np.max(batch.orig_F1s) if len(batch.orig_F1s) else 0.
                    eval_te_reward += te_acc
                    eval_te_boundary += np.max(batch.topic_F1s) if len(batch.topic_F1s) else 0.
                    nb_eval_examples += 1
                    if time > 0: nb_eval_steps += 1
                    time += 1

                batch.reset()

            result = {'training loss': tr_loss/np.max([nb_tr_examples, 1.e-10]),
                      'training te loss': tr_te_loss / np.max([nb_tr_steps, 1.e-10]),
                      'training te reward': (tr_te_reward / np.max([nb_tr_steps, 1.e-10]), tr_te_boundary / np.max([nb_tr_steps, 1.e-10])),
                      'training reward': (tr_reward/np.max([nb_tr_examples, 1.e-10]), tr_reward_boundary / np.max([nb_tr_examples, 1.e-10])),
                      'dev reward': (eval_reward/np.max([nb_eval_examples, 1.e-10]), eval_reward_boundary/np.max([nb_eval_examples, 1.e-10])),
                      'dev te reward': (eval_te_reward / np.max([nb_eval_steps, 1.e-10]), eval_te_boundary / np.max([nb_eval_steps, 1.e-10]))}

            eval_reward = eval_reward/np.max([nb_eval_examples, 1.e-10])

            if eval_reward >= max_eval_reward:
                max_eval_reward = eval_reward
                if args.do_eval == 2: test_instances = test_instances[:1]
                eval_reward, nb_eval_steps, nb_eval_examples, eval_pred_cps, eval_pred_top_ans, eval_reward_boundary = 0, 0, 0, [], [], 0
                if args.do_train: test_instances = test_instances[:10]
                if args.do_debug: test_instances = test_instances[1078:1079] #

                for eval_step in trange(np.min([3000, len(test_instances)]), desc="Test"): #len(test_instances)
                    batch = test_instances[eval_step]
                    if args.do_debug: print('\neval_step batch***', str(batch).encode('utf-8', 'ignore'))
                    time, reward, top_pred_ans = 0, [0], defaultdict(int)
                    te_acces = []

                    while time < len(batch.question):
                        update_train_instance(batch)
                        cp, ts, hn, mcl = retrieve_KB(batch, tokenizer, config, do_debug=args.do_debug,
                                                        time = time, is_train=False,
                                                        not_update = args.do_train)
                        q = batch.question[time]['question_idx']
                        te_logits, te_acc, max_te = None, 0, 0
                        if config.method in ['RecurrentRanker', 'SimpleRecurrentRanker'] and time > 0:
                            policy.ranker.forward_answer()

                            if config.method in ['RecurrentRanker']:
                                te_batch = select_te_field(batch, device)
                                _, te_acc, max_te = policy.ranker.predict_next_te(batch.topic_F1s, te_batch, time)

                                if time > 0 and config.method in ['RecurrentRanker']: #, 'RecurrentRanker_match'
                                    te_logits = policy.ranker.te_logits

                        te_acces += [te_acc]
                        ts_top = None #filter_ts(ts, te_logits, time=time)
                        ts = edit_ts(ts, te_logits, time=time) #te_logits batch.topic_F1s

                        if len(cp) == 0: break #time += 1; continue
                        ready_batch = select_field(q, cp, ts, hn, mcl, batch, config, ts_top=ts_top)
                        if args.gpu_id: ready_batch = tuple(t.to(device) for t in ready_batch)

                        # Step through environment using chosen action
                        with torch.no_grad():
                            if config.method in ['RecurrentRanker', 'SimpleRecurrentRanker']:
                                if time == 0: policy.ranker.initial_hidden_state()
                                _logits, _losses = policy.ranker(ready_batch, time)
                            else:
                                _logits, _ = policy.ranker(ready_batch, time)
                            _logits = F.softmax(_logits, 1)

                        logits = _logits.cpu().data.numpy() if args.gpu_id else _logits.data.numpy()
                        F1s = torch.tensor(batch.F1s, dtype=torch.float).view(1, -1)
                        if args.gpu_id: _F1s = F1s.to(device)

                        _action, _adjust_loss = select_action(policy, _logits, adjust_F1s=None,
                                                              is_train=False, time=time, is_reinforce=args.do_policy_gradient)
                        if config.method in ['RecurrentRanker', 'SimpleRecurrentRanker']:
                            policy.ranker.update_answer(_action)

                        action = _action.cpu().data.numpy() if args.gpu_id else _action.data.numpy()
                        reward, pred_cp, reward_boundary, pred_ans = generate_reward(logits, action, batch, do_debug=args.do_debug, eval_metric='F1')
                        if args.do_debug: print('pred_cp***', pred_cp, pred_ans, batch.question[time]['answer'], reward, reward_boundary)

                        pred_ans = str(np.max(batch.orig_F1s))
                        if config.method in ['RecurrentRanker']:
                            eval_pred_cp = re.sub('\n', '', '%s|%s\t%s\t%s\t%s\t%s' % (eval_step+1, time, str(max_te), str(pred_cp, 'utf-8'), reward, reward_boundary))
                        else:
                            eval_pred_cp = re.sub('\n', '', '%s|%s\t%s\t%s\t%s' %(eval_step+1, time, str(pred_cp, 'utf-8'), reward, reward_boundary))

                        eval_pred_cps += [eval_pred_cp.encode('utf-8', 'ignore').decode('utf-8')]
                        eval_pred_top_ans += [top_pred_ans]

                        eval_reward += np.mean(reward)
                        eval_reward_boundary += np.max(batch.orig_F1s)
                        nb_eval_examples += 1
                        nb_eval_steps += 1
                        time += 1

                    batch.reset()
                result['test reward'] = (eval_reward/np.max([nb_eval_examples, 1.e-10]), eval_reward_boundary / np.max([nb_eval_examples, 1.e-10]))
                if args.do_eval == 2: print(result); exit()

                g = open(save_eval_cp_file, "w", encoding='utf8',errors="ignore")
                g.write('\n'.join(eval_pred_cps))
                g.close()
                if args.do_debug: print(result); exit()

                if eval_pred_top_ans:
                    g = open(re.sub('.txt$', '.json', save_eval_cp_file), "w")
                    for top_pred_ans in eval_pred_top_ans:
                        json.dump(top_pred_ans, g)
                        g.write('\n')
                    g.close()

                if args.do_train:
                    '''save the model and some kb cache'''
                    model_to_save = policy.module if hasattr(policy, 'module') else policy
                    torch.save(model_to_save.state_dict(), save_model_file)


            with open(save_eval_file, "a") as writer:
                print('save to ', save_eval_cp_file)
                logger.info("***** Eval results (%s)*****" %epoch)
                writer.write("***** Eval results (%s)*****\n" %epoch)
                for key in sorted(result.keys()):
                    logger.info(" %s=%s", key, str(result[key]))
                    writer.write("%s=%s \n" %(key, str(result[key])))
            my_sparql.save_cache()

if __name__ == '__main__':
    main()
