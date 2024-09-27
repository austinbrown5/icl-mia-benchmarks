from datasets import load_dataset
import nltk
from nltk.tag import StanfordPOSTagger
import random
import os
import json
from prompts import Prompt
import time

def main():
    # Load dataset
    dataset_name = "meta-math/MetaMathQA"
    meta_math_ds = load_dataset(dataset_name, streaming=True)
    meta_math_ds = meta_math_ds.take(1000)
    meta_math_training = meta_math_ds['train']

    # Initialize Stanford POS tagger
    os.environ['CLASSPATH'] = "/usr/project/xtmp/arb153/icl-mia-benchmarks/stanford-postagger-full-2020-11-17/stanford-postagger.jar"
    os.environ["STANFORD_MODELS"] = "/usr/project/xtmp/arb153/icl-mia-benchmarks/stanford-postagger-full-2020-11-17/models"
    tagger = StanfordPOSTagger('english-bidirectional-distsim.tagger')

    # Initialize prompt data structures
    prompts = {
        "general_prompts": [],
        "guided_prompts": [],
        "ts_prompts": [],
        "standard_queries": []
    }
    targets = {
        "guided": [],
        "ts": [],
        "answers": []
    }

    # Generate prompts
    current_time = time.time()
    for data_point in meta_math_training:
        splits = guided_prompt_split_fn(data_point, 'query')
        guided_prompt_insert = splits['guided_prompt_part_1']
        guided_prompt_target = splits['guided_prompt_part_2']
        
        prompt = Prompt()
        guided_prompt = prompt.get_prompt("guided").format(dataset_name=dataset_name, first_piece=guided_prompt_insert)
        general_prompt = prompt.get_prompt("general").format(first_piece=guided_prompt_insert)
        ts_prompt, ts_target = ts_guessing_prompt(data_point, tagger, 'query')

        prompts['general_prompts'].append(general_prompt)
        prompts['guided_prompts'].append(guided_prompt)
        prompts['ts_prompts'].append(ts_prompt)
        prompts['standard_queries'].append(data_point['query'])
        
        targets['guided'].append(guided_prompt_target)
        targets['ts'].append(ts_target)
        targets['answers'].append(data_point['response'])

    print(f"Time to generate prompts: {time.time() - current_time}")
    # Save prompts and targets to files
    with open('prompts.json', 'w') as f:
        json.dump(prompts, f)
    with open('targets.json', 'w') as f:
        json.dump(targets, f)

if __name__ == "__main__":
    main()

def guided_prompt_split_fn(example, text_key):
    splits = {'guided_prompt_part_1': '', 'guided_prompt_part_2': ''}
    text = example[text_key]
    sentences = nltk.sent_tokenize(text)

    if len(sentences) == 1:
        # Split the single sentence in half
        mid_point = len(text) // 2
        splits['guided_prompt_part_1'] = text[:mid_point]
        splits['guided_prompt_part_2'] = text[mid_point:]
    elif len(sentences) > 1:
        # Original logic for multiple sentences
        first_part_length = random.randint(1, len(sentences) - 1)
        splits['guided_prompt_part_1'] = ' '.join(sentences[:first_part_length])
        splits['guided_prompt_part_2'] = ' '.join(sentences[first_part_length:])
    else:
        # Handle empty text case
        splits['guided_prompt_part_1'] = ''
        splits['guided_prompt_part_2'] = ''

    return splits

def ts_guessing_prompt(
    example, 
    tagger,
    text_key,
    type_hint=False,
):
    #question based prompt generation for ts guessing
    text = example[text_key]
    tags = tagger.tag(text.split())
    words = [x for x in tags if x[1] in ['NN', 'JJ', 'VB']]
    if len(words) == 0:
        return "failed", ""
    idx = np.random.randint(len(words))
    word = words[idx][0]
    for i in range(len(text)-len(word)+1):
        if text[i:(i+len(word))] == word:
            text = text[:i] + "[MASK]" + text[(i+len(word)):]
            break

    prompt = "Complete the sentence in one word:"
    prompt += f"\n\n{text}"
    if type_hint:
        example_type = example["type"]
        prompt += f"\nHint: {example_type}"
    prompt += "\nReply the answer only."

    return prompt, word