import os
import json
import re

class TrainingInstance(object):
    """A single training instance (A question associated with its topic entities)"""
    def __init__(self, q_idx, questions):
        self.q_idx = q_idx
        self.topic_entity_text = questions['seed_entity_text']
        self.topic_entity = questions['seed_entity']
        self.question = questions['questions']
        self.current_F1s = []
        self.orig_F1s = []
        self.F1s = []
        self.topic_F1s = []
        self.candidate_paths = []
        self.orig_candidate_paths = []
        self.statements = []
        self.start_statements = {}
        self.graph = []
        self.frontier2typeidx = []
        self.frontier2idx = {}
        self.historical_frontier = []
        self.current_topics = []
        self.node_attri = []
        self.current_frontier = []
        self.current_frontier_idx = []
        self.adjacent_matrix = []

    def __str__(self):
        s = ""
        s += "question: %s\n" %str([q['question'] for q in self.question])
        s += "answer: %s\n" % str([q['answer'] for q in self.question])
        s += "ner: %s\n" % str([q['ner'] for q in self.question])
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()

    def reset(self):
        self.frontier2typeidx = []
        self.frontier2idx = {}
        self.historical_frontier = []
        self.node_attri = []
        self.current_frontier = []
        self.current_frontier_idx = []
        self.adjacent_matrix = []
        self.current_topics = []
        self.graph = []


def create_instances(input_file, tokenizer):
    raw_questions, questions, topic_entities, answers, golden_graphs, constraints = [], [], [], [], [], []
    line_idx = 0
    category = os.path.basename(input_file)
    if 'CONVEX' in input_file:
        Q_file = json.load(open(os.path.join(input_file, 'processed_q.json'), 'rb'))
    elif 'CSQA' in input_file:
        Q_file = json.load(open(os.path.join(input_file, 'sequential_q_v2.json'), 'rb'))
    total_qas = []
    limit_num = -1 if 'train' in input_file else -1 if 'dev' in input_file else -1
    for line_idx, line in enumerate(Q_file):
        if line_idx == limit_num: break
        questions = line['questions']

        qas = {'questions': []}
        qas['seed_entity'] = [re.sub('^https\:\/\/www\.wikidata\.org\/wiki\/', '', line['seed_entity'])]
        #qas['seed_entity'] = tagme_get_all_entities(question, tagmeToken)
        qas['seed_entity_text'] = [line['seed_entity_text']] if ('seed_entity_text' in line) else []
        for q_idx, q in enumerate(questions):
            qa = {}
            qa['question'] = re.sub('\?$', '', q['question'].lower())
            #print("tokenizer.tokenize(q['question'])", tokenizer.tokenize(q['question']))
            qa['question_idx'] = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(q['question']))
            qa['answer'] = [re.sub('^[ ]*https\:\/\/www\.wikidata\.org\/wiki\/', '', a) for a in re.sub('\.$', '', q['answer']).split(';')]
            qa['ner'] = [re.sub('^[ ]*https\:\/\/www\.wikidata\.org\/wiki\/', '', w) for w in q['NER']]
            qa['answer_text'] = q['answer_text'].split(';')
            qa['relation'] = q['relation'] if 'relation' in q else ''
            qas['questions'] += [qa]

        #print('qas', qas); exit()
        total_qas += [qas]

    instances = []
    for q_idx, q in enumerate(total_qas):
        instances.append(TrainingInstance(q_idx=category + str(q_idx + 1),
                                        questions = total_qas[q_idx]))

    return instances