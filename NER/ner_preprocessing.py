import numpy as np
from itertools import groupby
from collections import defaultdict
from datasets import load_dataset

def preprocess_example(example):
    """
    This function recieves as input a dictionary containing the text and the ground truth annotation
    of named entities and return the pair of inputs in a format suitable to use in a prompt.
    """
    ner_tags_dict = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC',
                     8: 'I-MISC'}
    text = example['tokens']

    ner_tags = example['ner_tags']
    ner_tags = [ner_tags_dict[el].split('-')[-1] for el in ner_tags]
    gb_ner = groupby(ner_tags)
    groups = []

    for k, v in gb_ner:
        groups.append(list(v))

    ner_output_dict = defaultdict(list)

    offset = 0
    for group in groups:
        corresponding_text = text[offset:offset + len(group)]
        offset += len(group)
        key = group[0]
        if key != 'O':
            ner_output_dict[key].append(' '.join(corresponding_text))
    text = ' '.join(text)
    return text, ner_output_dict


def compose_prompt(target, n_shot_examples):
    """
    This function recieves a target element and n examples that serve as context for the assignment
    and outputs a prompt. 
    1)Examples are a zipped object with the first element
    being the text and the second element being a dict with ner tags and the corresponding parts of the text
    2)Target is a string
    """
    explanation = f'Retrieve the people, organizations and locations mentioned in the text below\n'

    examples = ''

    for text, ner_tags in n_shot_examples:
        examples += f'text:{text}\n'
        examples += f'Named Entities: '
        for k1, k2 in zip(["PER", "ORG", "LOC"], ["People", "Organizations", "Locations"]):
            v = ner_tags.get(k1)
            if v:
                v = ', '.join(v)
                examples += f'{k2} - {v}; '
        examples += '\n\n'

    completion_target = f'text:{target}\nNamed Entities:'
    return explanation + examples + completion_target


def create_context(dataset, n_examples=3, min_tokens=10, min_ner=1):
    filtered_data = list(
        filter(lambda x: len(x['tokens']) >= min_tokens and len(np.nonzero(x['ner_tags'])[0]) > min_ner, dataset))
    examples = []
    selected_idx = np.random.choice(np.arange(len(filtered_data)), size=n_examples, replace=False)
    for i in selected_idx:
        examples.append(filtered_data[i])
    return examples


def generate_prompt(target, n_shot, min_tokens, min_ner):
    annotated_data = load_dataset("conllpp", split='train')
    selected_examples = create_context(annotated_data, n_examples=n_shot, min_tokens=min_tokens, min_ner=min_ner)
    examples = selected_examples[:-1]
    
    texts, ner_outputs = [], []
    for ex in examples:
        text, ner_output = preprocess_example(ex)
        texts.append(text)
        ner_outputs.append(ner_output)
    n_shot_examples = zip(texts, ner_outputs)

    prompt = compose_prompt(target, n_shot_examples)
    return prompt

def parse_response(response):
    response =  response['choices'][0]['text'].split(';')
    results_dict = {}
    for section in response:
        if not section:
            continue
        k, v = section.split('-')
        results_dict[k.strip()] = [el.strip() for el in v.split(',')]
    return results_dict


