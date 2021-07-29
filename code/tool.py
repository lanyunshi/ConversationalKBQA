import re
import time as mytime
import numpy as np
from collections import defaultdict

const_minimax_dic = 'amount|number|how many|final|first|last|predominant|biggest|major|warmest|tallest|current|largest|most|newly|son|daughter' #


def addin_historical_frontier(batch, first_topic_entity, previous_topic_frontier, previous_ans_frontier, tokenizer):
    batch.node_attri = [tokenizer.vocab['[unused5]']]*(len(batch.historical_frontier)+len(first_topic_entity+previous_topic_frontier+previous_ans_frontier))
    batch.current_frontier = []
    for te in (first_topic_entity + previous_topic_frontier + previous_ans_frontier):
        if te not in batch.historical_frontier:
            batch.frontier2idx[te] = len(batch.historical_frontier)
            #m, _ = batch.adjacent_matrix.shape
            #print('batch.adjacent_matrix', batch.adjacent_matrix.shape, np.zeros((m, 1)).shape)
            #batch.adjacent_matrix = np.concatenate([batch.adjacent_matrix, np.zeros((m, 1))], 1)
            #print('batch.adjacent_matrix', batch.adjacent_matrix.shape, np.zeros((1, m+1)).shape)
            #batch.adjacent_matrix = np.concatenate([batch.adjacent_matrix, np.zeros((1, m+1))], 0)
            batch.historical_frontier += [te]
            batch.current_frontier += [te]
            #batch.adjacent_matrix[batch.frontier2idx[te], batch.frontier2idx[te]] = tokenizer.vocab['[unused2]']
            batch.adjacent_matrix += [(batch.frontier2idx[te], batch.frontier2idx[te], 0)]
            if te in previous_ans_frontier:
                #print('yeah')
                batch.adjacent_matrix += [(0, batch.frontier2idx[te], 1)]
                batch.adjacent_matrix += [(batch.frontier2idx[te], 0, 2)]
                #batch.adjacent_matrix[0, batch.frontier2idx[te]] = tokenizer.vocab['[unused3]']
                #batch.adjacent_matrix[batch.frontier2idx[te], 0] = tokenizer.vocab['[unused4]']
                for i in range(len(batch.historical_frontier)-2, 0, -1):
                    #print('test: ', batch.historical_frontier[i], batch.node_attri[i])
                    if batch.node_attri[i] != tokenizer.vocab['[unused7]']:
                        break
                    else:
                        batch.adjacent_matrix += [(i, batch.frontier2idx[te], 1)]
                        batch.adjacent_matrix += [(batch.frontier2idx[te], i, 2)]
                        #batch.adjacent_matrix[batch.frontier2idx[te], i] = tokenizer.vocab['[unused4]']
                        #batch.adjacent_matrix[i, batch.frontier2idx[te]] = tokenizer.vocab['[unused3]']
        if te in first_topic_entity:
            batch.node_attri[batch.frontier2idx[te]] = tokenizer.vocab['[unused6]']
        elif te in previous_topic_frontier:
            batch.node_attri[batch.frontier2idx[te]] = tokenizer.vocab['[unused7]']
        elif te in previous_ans_frontier:
            batch.node_attri[batch.frontier2idx[te]] = tokenizer.vocab['[unused8]']
    batch.node_attri = batch.node_attri[:len(batch.historical_frontier)]
    # print('batch.frontier2idx', batch.frontier2idx, 'batch.historical_frontier', batch.historical_frontier)
    # print('first_topic_entity', first_topic_entity)
    # print('previous_topic_frontier', previous_topic_frontier)
    # print('previous_ans_frontier', previous_ans_frontier)
    # print('batch.historical_frontier', batch.historical_frontier)
    # print('batch.node_attri', batch.node_attri)
    # print('batch.adjacent_matrix', batch.adjacent_matrix)
    # print('batch.current_frontier', batch.current_frontier)

def retrieve_via_frontier(frontier, raw_candidate_paths, my_sparql, question=None, not_update=True):
    '''Single entity involves in a question'''
    not_update = False
    #print('frontier', frontier)
    for t in set(frontier):
        if not re.search('^Q', t): continue
        #print('my_sparql.STATEMENTS[(t, None)]')
        time1 = mytime.time()
        if (t, None) not in my_sparql.STATEMENTS:
            if not_update:
                statements = {}
            else:
                statements, sparql_txts = my_sparql.SQL_1hop(((t,),), my_sparql.QUERY_TXT)
                my_sparql.QUERY_TXT = my_sparql.QUERY_TXT.union(sparql_txts)
                statements_tmp, sparql_txts = my_sparql.SQL_2hop(((t,),), my_sparql.QUERY_TXT)
                my_sparql.QUERY_TXT = my_sparql.QUERY_TXT.union(sparql_txts)
                statements.update(statements_tmp)
                my_sparql.STATEMENTS[(t, None)].update(statements)
                #print('my_sparql.STATEMENTS[(t, None)]', mytime.time()-time1)
        else:
            statements = my_sparql.STATEMENTS[(t, None)]
            #print('cache my_sparql.STATEMENTS[(t, None)]', mytime.time()-time1)
        if statements: raw_candidate_paths += [statements]

        '''Multiple entities involve in a question'''
        #print('my_sparql.STATEMENTS[(t, sorted_topic_entity)]')
        sorted_topic_entity = tuple(sorted(set(frontier) - set([t])))
        #print(sorted_topic_entity)
        if False: #len(sorted_topic_entity):
            #print('sorted_topic_entity', sorted_topic_entity)
            time1 = mytime.time()
            if (t, sorted_topic_entity) not in my_sparql.STATEMENTS:
                if not_update:
                    statements = {}
                else:
                    statements, sparql_txts = my_sparql.SQL_2hop_reverse(((t,),), set(frontier) - set([t]))
                    my_sparql.QUERY_TXT = my_sparql.QUERY_TXT.union(sparql_txts)
                    my_sparql.STATEMENTS[(t, sorted_topic_entity)].update(statements)
                #print('my_sparql.STATEMENTS[(t, sorted_topic_entity)]', mytime.time()-time1)
            else:
                statements = my_sparql.STATEMENTS[(t, sorted_topic_entity)]
                #print('cache my_sparql.STATEMENTS[(t, sorted_topic_entity)]', mytime.time()-time1)
            if statements: raw_candidate_paths += [statements]

        #print('my_sparql.STATEMENTS[(t, const_type)], mid')
        if question is not None and re.search(const_minimax_dic, question):
            const_type = tuple(re.findall(const_minimax_dic, question))
            time1 = mytime.time()
            if (t, const_type) not in my_sparql.STATEMENTS:
                if not_update:
                    statements = {}
                else:
                    statements, sparql_txts = my_sparql.SQL_1hop_reverse(((t,),), const_type)
                    my_sparql.QUERY_TXT = my_sparql.QUERY_TXT.union(sparql_txts)
                    my_sparql.STATEMENTS[(t, const_type)].update(statements)
                #print('my_sparql.STATEMENTS[(t, const_type)], mid', mytime.time()-time1)
            else:
                statements = my_sparql.STATEMENTS[(t, const_type)]
                #print('cache my_sparql.STATEMENTS[(t, const_type)], mid', mytime.time()-time1)
            if statements: raw_candidate_paths += [statements]

        #print('my_sparql.STATEMENTS[(t, const_type)], year')
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

def generate_F1_tmp(pred_ans, ans):
    TP = len(set(pred_ans) & set(ans))
    precision = TP*1./np.max([len(set(pred_ans)), 1e-10])
    recall = TP*1./np.max([len(set(ans)), 1e-10])
    F1 = 2. * precision * recall/np.max([(precision + recall), 1e-10])
    return F1

def convert_json_to_save(dic):
    '''If dictionary is {keys1(tuple): keys2(tuple): value(set)}'''
    new_dic = {}
    for hs in dic:
        new_h = ' '.join([' '.join(list(r)) if isinstance(r, tuple) else str(r) for r in hs]) if isinstance(hs, tuple) else hs
        if len(dic[hs]) == 0: new_dic[new_h] = {}
        for rs in dic[hs]:
            new_r = '\t'.join([' '.join(r) for r in rs])
            new_t = list(dic[hs][rs])
            if new_h not in new_dic:
                new_dic[new_h] = {new_r: new_t}
            elif new_r not in new_dic[new_h]:
                new_dic[new_h][new_r] = new_t
    return new_dic

def convert_json_to_load(dic):
    new_dic = defaultdict(dict)
    for hs in dic:
        new_h = hs #tuple([tuple(r.split(' ')) for r in hs.split('\t')])
        if len(dic[hs]) == 0: new_dic[new_h] = {}
        for rs in dic[hs]:
            new_r = tuple([tuple(r.split(' ')) for r in rs.split('\t')])
            new_t = set(dic[hs][rs])
            if new_h not in new_dic:
                new_dic[new_h] = {new_r: new_t}
            elif new_r not in new_dic[new_h]:
                new_dic[new_h][new_r] = new_t
    return new_dic