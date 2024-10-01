from vllm import LLM, SamplingParams
import nltk
import random
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from scipy import stats
from rouge_score import rouge_scorer
from bleurt import score as bleurt_score
import json


def main():
    # Usage
    model_name = "meta-math/MetaMath-Mistral-7B"

    # llm = LLM(model=model_name, dtype="half")
    llm = LLM(
        model=model_name,
        dtype="half",
        max_model_len=2048,
        enforce_eager=True,
        max_num_seqs = 50,
        ) #decrease model max length to fit in kv cache for vllm
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95)
    
    #going to go through the dataset and generate our unique prompts, do not need these for our min_k, perplexity, or cdd attacks
    print('Loading Prompts...')
    with open('prompts/MetaMathQA/test_prompts.json', 'r') as f:
        test_prompts = json.load(f)
    with open('prompts/MetaMathQA/test_targets.json', 'r') as f:
        test_targets = json.load(f)
    with open('prompts/MetaMathQA/training_prompts.json', 'r') as f:
        training_prompts = json.load(f)
    with open('prompts/MetaMathQA/training_targets.json', 'r') as f:
        training_targets = json.load(f)

    prompts = merge_dicts(training_prompts, test_prompts)
    targets = merge_dicts(training_targets, test_targets)

    print('Performing Guided Prompting attack...')
    test_size = 30#how many samples to use in our significance testing
    gp_sampling_params = SamplingParams(temperature = 0.7, n = test_size, top_p = 0.95)

    general_outputs = llm.generate(prompts['general_prompts'], gp_sampling_params)
    guided_outputs = llm.generate(prompts['guided_prompts'], gp_sampling_params)

    rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    bleurt_scorer = bleurt_score.BleurtScorer()

    rouge_general_scores = [scorer.score(ref, hyp)['rougeL'].fmeasure for ref, hyp in zip(targets['answers'], [output.outputs[0].text for output in general_outputs])]
    rouge_guided_scores = [scorer.score(ref, hyp)['rougeL'].fmeasure for ref, hyp in zip(targets['guided'], [output.outputs[0].text for output in guided_outputs])]
    
    bleurt_general_scores = bleurt_scorer.score(references=targets['answers'], candidates=[output.outputs[0].text for output in general_outputs])
    bleurt_guided_scores = bleurt_scorer.score(references=targets['guided'], candidates=[output.outputs[0].text for output in guided_outputs])

    gp_scores = [guided - general for guided, general in zip(guided_scores, general_scores)]


    print('Performing TS Guessing attack...')
    ts_outputs = llm.generate(prompts['ts_prompts'], sampling_params)        
    ts_scores = [1 if output.outputs[0].text.strip().lower() == target.strip().lower() else 0 
                 for output, target in zip(ts_outputs, targets['ts'])]

    print('Performing CDD attack...')
    cdd_scores = cdd(prompts= prompts['standard_queries'], llm= llm, alpha = 0.05, xi = 0.01, num_samples = 20 )
    
    print('Performing Min K attack...')
    min_k_scores, loss_scores = min_k_loss(prompts['standard_queries'], llm, k_percent= 10)

    training_truths = np.ones(1000) # hard coded bc of debugging
    test_truths = np.zeros(1000) # hard coded bc of debugging

    truths = np.concatenate((training_truths, test_truths))
    print(len(cdd_scores))
    print(len(min_k_scores))
    print(len(loss_scores))
    print(len(gp_scores))
    print(len(ts_scores))
    assert len(truths) == len(cdd_scores)
    assert len(truths) == len(min_k_scores)
    assert len(truths) == len(loss_scores)
    assert len(truths) == len(gp_scores)
    assert len(truths) == len(ts_scores)

    #scoring
    aucroc_guided = roc_auc_score(truths, gp_scores)
    aucroc_ts = roc_auc_score(truths, ts_scores)
    aucroc_cdd = roc_auc_score(truths, cdd_scores)
    aucroc_min_k = roc_auc_score(truths, min_k_scores)

    print(f"AUCROC Guided: {aucroc_guided}")
    print(f"AUCROC TS: {aucroc_ts}")
    print(f"AUCROC CDD: {aucroc_cdd}")
    print(f"AUCROC Min-k: {aucroc_min_k}")

    for fpr_threshold in [0.01, 0.05, 0.10, 0.25]:
        tpr_guided = tpr_at_fpr(truths, gp_scores, fpr_threshold)
        tpr_ts = tpr_at_fpr(truths, ts_scores, fpr_threshold)
        tpr_cdd = tpr_at_fpr(truths, cdd_scores, fpr_threshold)
        tpr_min_k = tpr_at_fpr(truths, min_k_scores, fpr_threshold)

        print(f"TPR@{fpr_threshold * 100}%FPR Guided: {tpr_guided}")
        print(f"TPR@{fpr_threshold * 100}%FPR TS: {tpr_ts}")
        print(f"TPR@{fpr_threshold * 100}%FPR CDD: {tpr_cdd}")
        print(f"TPR@{fpr_threshold * 100}%FPR Min-k: {tpr_min_k}")

def tpr_at_fpr(y_true, y_score, fpr_threshold):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    index = next(i for i, x in enumerate(fpr) if x >= fpr_threshold)
    return tpr[index]

def guided_prompt_split_fn(example, text_key):
    splits = {'guided_prompt_part_1': '', 'guided_prompt_part_2': ''}
    text = example[text_key]
    sentences = nltk.sent_tokenize(text, )

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
    
def min_k_loss(texts, llm, k_percent=10):
    sampling_params = SamplingParams(temperature=0.0, logprobs=1, prompt_logprobs=1)
    outputs = llm.generate(texts, sampling_params)

    min_ks = []
    losses = []

    for output in outputs:
        prompt_logprobs = output.prompt_logprobs
        if prompt_logprobs is None or len(prompt_logprobs) == 0:
            print(f"Warning: No prompt logprobs for output: {output.text[:50]}...")
            min_ks.append(0)  # or some default value
            losses.append(0)  # or some default value
            continue

        logprobs = []
        for token_logprobs in prompt_logprobs:
            if token_logprobs is not None:
                # Get the highest logprob for each token
                max_logprob = max(lp.logprob for lp in token_logprobs.values())
                logprobs.append(max_logprob)

        if len(logprobs) == 0:
            print(f"Warning: No valid logprobs for output: {output.text[:50]}...")
            min_ks.append(0)  # or some default value
            losses.append(0)  # or some default value
            continue

        logprobs = np.array(logprobs)
        k = max(1, int(len(logprobs) * k_percent / 100))
        topk = np.sort(logprobs)[:k]
        min_k_score = np.mean(topk).item()
        min_ks.append(min_k_score)

        loss = -np.sum(logprobs)
        losses.append(loss)

    return min_ks, losses

# def cdd(prompts, llm, alpha = 0.05, xi = 0.01):
#     sampling_params_multiple = SamplingParams(
#         temperature=0.8,
#         n=50
#     )
#     sampling_params_greedy = SamplingParams(
#         temperature=0,
#         n=1
#     )
#     cdd_scores = []
#     for prompt in prompts:
#         samples = llm.generate([prompt] * 50, sampling_params_multiple)
#         generated_texts = [output.outputs[0].text for output in samples]

#         greedy_sample = llm.generate([prompt], sampling_params_greedy)[0]
#         s_0 = greedy_sample.outputs[0].text

#         peak = get_peak(generated_texts, s_0, alpha)
#         is_contaminated = peak / len(generated_texts) > xi
#         if is_contaminated:
#             cdd_scores.append(1)
#         else:
#             cdd_scores.append(0)
#     return cdd_scores
def cdd(prompts, llm, alpha=0.05, xi=0.01, num_samples = 20):
    #* num_samples normally higher, but set to 10 to speed up debugging
    sampling_params_multiple = SamplingParams(
        temperature=0.8,
        n=num_samples
    )
    sampling_params_greedy = SamplingParams(
        temperature=0,
        n=1
    )
    
    # Generate all samples at once
    all_samples = llm.generate(prompts, sampling_params_multiple)
    greedy_samples = llm.generate(prompts, sampling_params_greedy)

    cdd_scores = []
    for i, prompt in enumerate(prompts):
        # Extract the 50 samples for this prompt
        generated_texts = [output.text for output in all_samples[i].outputs]
        # generated_texts = [all_samples[i* num_samples + j].outputs[0].text for j in range(num_samples)]
        
        # Get the greedy sample for this prompt
        s_0 = greedy_samples[i].outputs[0].text

        peak = get_peak(generated_texts, s_0, alpha)
        is_contaminated = peak / len(generated_texts) > xi
        # cdd_scores.append(1 if is_contaminated else 0)
        cdd_scores.append(peak)

    return cdd_scores

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def get_peak(samples, s_0, alpha):
    lengths = [len(x) for x in samples]
    l = min(lengths)
    l = min(l, 100)
    thresh = int(np.ceil(alpha * l))
    distances = [levenshtein_distance(s, s_0) for s in samples]
    rhos = [len([x for x in distances if x == d]) for d in range(0, thresh+1)]
    peak = sum(rhos)

    return peak

def merge_dicts(dict1, dict2):
    return {key: dict1[key] + dict2[key] for key in dict1}

if __name__ == "__main__":
    main()
