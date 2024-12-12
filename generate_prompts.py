from datasets import load_dataset
import nltk
from nltk.tag import StanfordPOSTagger
import random
import os
import json
from prompts import Prompt
import time
import ssl
import numpy as np

def main():
    # Load dataset
    dataset_name = "meta-math/MetaMathQA"
    test_dataset_name = "meta-math/GSM8K_Backward"

    # Load openai key (for ts-guessing icl)
    chatgpt_api_key = os.getenv("OPENAI_API_KEY")
    assert chatgpt_api_key

    # meta_math_ds = load_dataset(dataset_name, streaming=True)
    # gsm8k_b = load_dataset(test_dataset_name, streaming =True)

    # meta_math_training = meta_math_ds['train'].take(1000)
    # meta_math_test = gsm8k_b['test'].take(1000)

    meta_math_training = load_json_data('data/train.jsonl', num_samples = 1000)
    meta_math_test = load_json_data('data/test.jsonl', num_samples = 1000)

    # # Initialize Stanford POS tagger
    # os.environ['CLASSPATH'] = "/usr/project/xtmp/arb153/icl-mia-benchmarks/stanford-postagger-full-2020-11-17/stanford-postagger.jar"
    # os.environ["STANFORD_MODELS"] = "/usr/project/xtmp/arb153/icl-mia-benchmarks/stanford-postagger-full-2020-11-17/models"
    # tagger = StanfordPOSTagger('english-bidirectional-distsim.tagger')
    import nltk
    nltk.download('averaged_perceptron_tagger_eng')
    tagger = nltk.pos_tag

    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download('punkt_tab')

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

    test_prompts = {
        "general_prompts": [],
        "guided_prompts": [],
        "ts_prompts": [],
        "standard_queries": []
    }
    test_targets = {
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
        ts_prompt, ts_target = ts_guessing_prompt(data_point, tagger, 'query', chatgpt_api_key = chatgpt_api_key)

        prompts['general_prompts'].append(general_prompt)
        prompts['guided_prompts'].append(guided_prompt)
        prompts['ts_prompts'].append(ts_prompt)
        prompts['standard_queries'].append(data_point['query'])

        targets['guided'].append(guided_prompt_target)
        targets['ts'].append(ts_target)
        targets['answers'].append(data_point.response)

    print(f"Time to generate prompts: {time.time() - current_time}")
    # Save prompts and targets to files
    with open('training_prompts.json', 'w') as f:
        json.dump(prompts, f)
    with open('training_targets.json', 'w') as f:
        json.dump(targets, f)

    current_time = time.time()
    for data_point in meta_math_test:
        splits = guided_prompt_split_fn(data_point, 'query')
        guided_prompt_insert = splits['guided_prompt_part_1']
        guided_prompt_target = splits['guided_prompt_part_2']

        prompt = Prompt()
        guided_prompt = prompt.get_prompt("guided").format(dataset_name=dataset_name, first_piece=guided_prompt_insert)
        general_prompt = prompt.get_prompt("general").format(first_piece=guided_prompt_insert)
        ts_prompt, ts_target = ts_guessing_prompt(data_point, tagger, 'query', chatgpt_api_key = chatgpt_api_key)

        test_prompts['general_prompts'].append(general_prompt)
        test_prompts['guided_prompts'].append(guided_prompt)
        test_prompts['ts_prompts'].append(ts_prompt)
        test_prompts['standard_queries'].append(data_point['query'])

        test_targets['guided'].append(guided_prompt_target)
        test_targets['ts'].append(ts_target)
        test_targets['answers'].append(data_point.response)

    print(f"Time to generate test prompts: {time.time() - current_time}")
    # Save prompts and targets to files
    with open('test_prompts.json', 'w') as f:
        json.dump(prompts, f)
    with open('test_targets.json', 'w') as f:
        json.dump(targets, f)

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

from openai import OpenAI

client = OpenAI()

def ts_guessing_prompt(
    example, 
    tagger,
    text_key,
    type_hint=False,
    chatgpt_api_key=None,
    num_shots=5
):
    if chatgpt_api_key is None:
        raise ValueError("ChatGPT API key is required.")
    text = example[text_key]
    tags = tagger(text.split())
    words = [x for x in tags if x[1] in ['NN', 'JJ', 'VB']]

    if len(words) == 0:
        return "failed", ""

    few_shot_examples = [
        "Q: What is the main event in the sentence 'The quick brown fox jumps over the lazy dog'?\nA: jumps",
        "Q: What is the main characteristic of the fox in 'The quick brown fox jumps over the lazy dog'?\nA: quick",
        "Q: What is the key entity in 'The meeting was held in New York City'?\nA: meeting",
        "Q: What is the key entity in 'The cat sat on the mat'?\nA: cat",
        "Q: What is the key action in 'She quickly finished her homework'?\nA: finished"
    ]

    selected_few_shots = "\n\n".join(few_shot_examples[:num_shots])

    prompt = f"{selected_few_shots}\n\nQ: What is the most significant word in the sentence '{text}'?\nA:"

    try:
        response = client.chat.completions.create(model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5,
        temperature=0.3)
        most_significant_word = response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error with ChatGPT API: {e}", ""

    if most_significant_word not in [x[0] for x in words]:
        return "failed", ""

    word = most_significant_word
    for i in range(len(text) - len(word) + 1):
        if text[i:(i + len(word))] == word:
            text = text[:i] + "[MASK]" + text[(i + len(word)):]
            break

    final_prompt = "Complete the sentence in one word:"
    final_prompt += f"\n\n{text}"
    if type_hint:
        example_type = example.get("type", "unknown")
        final_prompt += f"\nHint: {example_type}"
    final_prompt += "\nReply the answer only."

    return final_prompt, word

def load_json_data(file_path, num_samples=1000):
    data = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            data.append(json.loads(line.strip()))
    return data

if __name__ == "__main__":
    main()