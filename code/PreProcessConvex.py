import os
import json
import re

from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
from SPARQL_test import sparql_test

def generate_NER_tags(sentence, predictor):
    result = predictor.predict(sentence=sentence)
    NER_span, token = [], []
    for w_idx, w in enumerate(result['words']):
        if result['tags'][w_idx] == 'O' and len(token):
            NER_span += [token]
            token = []
        elif result['tags'][w_idx] != 'O':
            token += [w]
    if len(token): NER_span += [token]
    return NER_span

def generate_CONVEX_file(data_path, category, predictor):
    Q_file = json.load(open(os.path.join(data_path, '%s_set' %category, '%s_set_ALL.json' %category), 'rb'))

    lines = []
    for line_idx, line in enumerate(Q_file):
        #if line_idx == 10: break
        if (line_idx+1)%100==0: print(category, line_idx)
        for question in line['questions']:
            q = question['question']
            ners = generate_NER_tags(q, predictor)
            ners = [' '.join(e) for e in ners]
            question['NER'] = ners
            #print(question); exit()
        lines += [line]

    g = open(os.path.join(data_path, '%s_set' %category, 'processed_q.json'), 'w')
    json.dump(lines, g)
    g.close()

def generate_CONVEX_for_D2A(data_path, category):
    predicate_out = []
    with open('/home/yslan/Dialog-to-Action/data/RC/vocab.out', 'rb') as f:
        for line_idx, line in enumerate(f):
            predicate_out += [line.strip()]

    with open('/home/yslan/MaSP/data/BFS/test/QA12.json') as f:
        lines = json.load(f)
    #print(lines[0], lines[1])

    if not os.path.exists(os.path.join(data_path, '%s' %category)):
        os.makedirs(os.path.join(data_path, '%s' %category))

    my_sparql = sparql_test()
    cache_dir = '/home/yslan/LargeCache/KBQA/CSQA_v1'
    # my_sparql.load_cache('%s/M2N.json' % cache_dir,
    #                      '%s/STATEMENTS.json' % cache_dir,
    #                      '%s/QUERY.json' % cache_dir,
    #                      '%s/TYPE.json' % cache_dir,
    #                      '%s/OUTDEGREE.json' % cache_dir)
    with open(os.path.join(data_path, '%s_set' %category, 'processed_q.json')) as f:
        lines = json.load(f)
        for line_idx, line in enumerate(lines):
            #if line_idx == 10: break
            output = []
            filename = 'QA' + str(line['conv_id'])
            for l in line['questions']:
                #print(l); #exit()
                q = {'ques_type_id': 1, 'question-type': 'Simple Question (Direct)', 'description': 'Simple Question'}
                ner = [my_sparql.wikidata_label_to_id(w) for w in l['NER']]
                q['entities_linking'] = [w for w in ner if w != 'UNK']
                q['utterance'] = l['question']
                q['relations'] = []
                q['type_list'] = []
                q['speaker'] = 'USER'
                q['entities_in_utterance'] = [w for w in ner if w != 'UNK']
                q['predicate_prediction'] = predicate_out
                output += [q]
                q = {'ques_type_id': 1, 'question-type': 'Simple Question (Direct)', 'description': 'Simple Question'}
                ner = [my_sparql.wikidata_label_to_id(w) for w in l['NER']]
                q['entities_linking'] = [w for w in ner if w != 'UNK']
                q['utterance'] = ', '.join(l['answer_text'].split(';'))
                q['active_set'] = []
                q['all_entities'] = re.findall('Q\d+', l['answer'])
                q['speaker'] = 'SYSTEM'
                q['entities_in_utterance'] = re.findall('Q\d+', l['answer'])
                q['predicate_prediction'] = predicate_out
                output += [q]
            #print('output', output); exit()
            g = open(os.path.join(data_path, '%s' %category, filename+'.json'), 'w')
            json.dump(output, g)
            g.close()
            #exit()


data_path = "CONVEX/data"
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/ner-model-2020.02.10.tar.gz")

for category in ['test', 'dev', 'train']:
    generate_CONVEX_file(data_path, category, predictor)
    #generate_CONVEX_for_D2A(data_path, category)

'''
python code/PreProcessConvex.py
'''