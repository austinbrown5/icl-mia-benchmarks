from vllm import LLM, SamplingParams
import nltk
import random
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from scipy import stats
from rouge_score import rouge_scorer
from bleurt import score as bleurt_score
import json
import logging
from openai import OpenAI
import pandas as pd
import json
from collections import defaultdict
import os
from transformers import AutoTokenizer


def main():
    logging.basicConfig(filename='ts_guessing_attack.log', level=logging.INFO, format='%(message)s')
    # Usage
    #model_name = "neuralmagic/Llama-2-7b-gsm8k"
    model_name = "meta-math/MetaMath-Mistral-7B"
    llm = LLM(
        model=model_name,
        tensor_parallel_size=2,  # Use both GPUs
        device = "cuda",
        dtype="auto",  # Use float16 instead of bfloat16 for better memory efficiency
    )
    # sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    
    #going to go through the dataset and generate our unique prompts, do not need these for our min_k, perplexity, or cdd attacks
    print('Loading Prompts...')
    # with open('prompts/MetaMathQA/test_prompts.json', 'r') as f:
    #     test_prompts = json.load(f)
    # with open('prompts/MetaMathQA/test_targets.json', 'r') as f:
    #     test_targets = json.load(f)
    # with open('prompts/MetaMathQA/training_prompts.json', 'r') as f:
    #     training_prompts = json.load(f)
    # with open('prompts/MetaMathQA/training_targets.json', 'r') as f:
    #     training_targets = json.load(f)
    # with open('data/truthful_qa_our/training_prompts_tqa.json', 'r') as f:
    #     training_prompts = json.load(f)
    # with open('data/truthful_qa_our/training_targets_tqa.json', 'r') as f:
    #     training_targets = json.load(f)

    #! SUBSET ITERATIVE NGRAM
    # base_path = "data/gsm8k/rewritten"
    # train_data = defaultdict(lambda: {'original': None, 'variations': []})
    # test_data = defaultdict(lambda: {'original': None, 'variations': []})

    # # Set desired subset size
    # SUBSET_SIZE = 50

    # # Read all files
    # for split in ['train', 'test']:
    #     data_dict = train_data if split == 'train' else test_data
        
    #     # Get count from first variation file
    #     filename = f"GSM8K_rewritten-{split}-1.jsonl"
    #     filepath = os.path.join(base_path, filename)
        
    #     # Read only SUBSET_SIZE lines
    #     count = 0
    #     with open(filepath, 'r', encoding='utf-8') as f:
    #         for _ in range(SUBSET_SIZE):
    #             count += 1
                
    #     if split == 'train':
    #         train_count = count
    #     else:
    #         test_count = count
        
    #     for variation in range(1, 4):
    #         filename = f"GSM8K_rewritten-{split}-{variation}.jsonl"
    #         filepath = os.path.join(base_path, filename)
            
    #         with open(filepath, 'r', encoding='utf-8') as f:
    #             for idx, line in enumerate(f):
    #                 # Break after reaching subset size
    #                 if idx >= SUBSET_SIZE:
    #                     break
                        
    #                 data = json.loads(line)
                    
    #                 # Store original question only once
    #                 if variation == 1:
    #                     data_dict[idx]['original'] = {
    #                         'question': data['question'],
    #                         'answer': data['answer']
    #                     }
                    
    #                 # Store rewritten version
    #                 data_dict[idx]['variations'].append({
    #                     'question': data['rewritten_question'],
    #                     'answer': data['rewritten_answer'],
    #                     'variation': variation
    #                 })

    
    train_data = read_json_file("processed_metamath_test_50.json")
    test_data = read_json_file("processed_metamath_test_50.json")

    n_train = len(train_data)
    n_test = len(test_data)

    print('Performing Iterative N-Gram accuracy attack...')
    train_n_grams = defaultdict()
    test_n_grams = defaultdict()

    train_ts_prompts = []
    train_ts_answers = []
    test_ts_prompts = []
    test_ts_answers = []

    for i, point in enumerate(train_data):
        train_n_grams[i] = point['n_gram']
        train_ts_prompts.append(point["ts"]["masked"])
        train_ts_answers.append(point["ts"]["mask"])

    for i,point in enumerate(test_data):
        test_n_grams[i] = point['n_gram']
        test_ts_prompts.append(point["ts"]["masked"])
        train_ts_answers.append(point["ts"]["mask"])

    train_n_gram_scores = calculate_n_gram_accuracy(5, train_n_grams, llm)
    test_n_gram_scores = calculate_n_gram_accuracy(5, test_n_grams, llm)
    n_gram_scores = np.concatenate((train_n_gram_scores, test_n_gram_scores))

#     base_path = "data/gsm8k/rewritten"
#     train_data = defaultdict(lambda: {'original': None, 'variations': []})
#     test_data = defaultdict(lambda: {'original': None, 'variations': []})

#     # Counters for tracking
#     train_count = 0
#     test_count = 0

#     # Read all files
#     for split in ['train', 'test']:
#         data_dict = train_data if split == 'train' else test_data
        
#         # Get count from first variation file (since all variations have same number of examples)
#         filename = f"GSM8K_rewritten-{split}-1.jsonl"
#         filepath = os.path.join(base_path, filename)
#         with open(filepath, 'r', encoding='utf-8') as f:
#             count = sum(1 for line in f)
#         if split == 'train':
#             train_count = count
#         else:
#             test_count = count
        
#         for variation in range(1, 4):
#             filename = f"GSM8K_rewritten-{split}-{variation}.jsonl"
#             filepath = os.path.join(base_path, filename)
            
#             with open(filepath, 'r', encoding='utf-8') as f:
#                 for idx, line in enumerate(f):
#                     data = json.loads(line)
                    
#                     # Store original question only once
#                     if variation == 1:
#                         data_dict[idx]['original'] = {
#                             'question': data['question'],
#                             'answer': data['answer']
#                         }
                    
#                     # Store rewritten version
#                     data_dict[idx]['variations'].append({
#                         'question': data['rewritten_question'],
#                         'answer': data['rewritten_answer'],
#                         'variation': variation
#                     })

#     # this is authors data for ts guessing on truthful qa
#     # fill_in = pd.read_csv("/usr/project/xtmp/arb153/icl-mia-benchmarks/data/truthful_qa_ts/fill_in_mc1.csv", index_col = 0)
#     # icl_mask = pd.read_csv("/usr/project/xtmp/arb153/icl-mia-benchmarks/data/truthful_qa_ts/icl_tagging.csv", index_col= 0)
#     # prompts = merge_dicts(training_prompts, test_prompts)
#     # targets = merge_dicts(training_targets, test_targets)
#     # prompts = training_prompts
#     # targets = training_targets

    # print('Performing Guided Prompting attack...')
    # test_size = 150#how many samples to use in our significance testing
    # gp_sampling_params = SamplingParams(temperature = 0.25, n = test_size, top_p = 0.95)
    # # gp_bootstrap_resampling = SamplingParams(temperature = 0, n = 1, top_p = 0.95)

    # general_outputs = llm.generate(prompts['general_prompts'], gp_sampling_params)
    # guided_outputs = llm.generate(prompts['guided_prompts'], gp_sampling_params)
    # # bootstrap_general_outputs = llm.generate(prompts['general_prompts'], gp_bootstrap_resampling)
    # # bootstrap_guided_outputs = llm.generate(prompts['guided_prompts'], gp_bootstrap_resampling)

    # #general outputs and guided outputs will contain test_size entries per prompt
    # #for each of these prompts, we want to calculate the rouge and bleurt scores compared to the target
    # #this means we will have 30 rouge scores and 30 bleurt scores per prompt for guided and general
    # #we then want to conduct a signifigance test to see if there is a difference in scores for guided or general
    # #i should now have two p values per target, one for the test of the bleurt scores and one for the rouge l scores
    # rs = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    # bleurt_scorer = bleurt_score.BleurtScorer()

    # gp_scores = []

    # rouge_scores_general = []
    # rouge_scores_guided = []
    # bleurt_scores_general = []
    # bleurt_scores_guided = []

    # # bootstrap_genr = []
    # # bootstrap_gr = []
    # # boostrap_genb = []
    # # bootstrap_gb = []

    # # for i, target in enumerate(targets['guided']):
    # #     gen = general_outputs[i].outputs[0].text
    # #     guided = guided_outputs[i].outputs[0].text

    # #     bsgeneral_rouge = rs.score(target, gen)['rougeL'].fmeasure
    # #     bsgeuided_rouge = rs.score(target, guided)['rougeL'].fmeasure

    # #     bs_guided_bleurt =  bleurt_scorer.score(references = [target], candidates = [gen])[0]
    # #     bs_general_bleurt = bleurt_scorer.score(references = [target], candidates = [guided])[0]

    # #     bootstrap_genr.append(bsgeneral_rouge)
    # #     bootstrap_gr.append(bsgeuided_rouge)
    # #     boostrap_genb.append(bs_general_bleurt)
    # #     bootstrap_gb.append(bs_guided_bleurt)

    # for i, target in enumerate(targets['guided']):
    #     general_rouge_scores = []
    #     general_bleurt_scores = []
    #     guided_rouge_scores = []
    #     guided_bleurt_scores = []

    #     assert len(general_outputs[i].outputs) == test_size, 'LLM Generation failed, we do not have n generations per prompt'
    #     assert len(guided_outputs[i].outputs) == test_size, 'LLM Generation failed, we do not have n generations per prompt'
        
    #     for j in range(test_size):
    #         general_output = general_outputs[i].outputs[j].text
    #         guided_output = guided_outputs[i].outputs[j].text
            
    #         general_rouge = rs.score(target, general_output)['rougeL'].fmeasure
    #         guided_rouge = rs.score(target, guided_output)['rougeL'].fmeasure
            
    #         general_bleurt = bleurt_scorer.score(references = [target], candidates = [general_output])[0]
    #         guided_bleurt = bleurt_scorer.score(references = [target], candidates = [guided_output])[0]
            
    #         general_rouge_scores.append(general_rouge)
    #         general_bleurt_scores.append(general_bleurt)
    #         guided_rouge_scores.append(guided_rouge)
    #         guided_bleurt_scores.append(guided_bleurt)
        
    #     rouge_scores_general.append(general_rouge_scores)
    #     rouge_scores_guided.append(guided_rouge_scores)
    #     bleurt_scores_general.append(general_bleurt_scores)
    #     bleurt_scores_guided.append(guided_bleurt_scores)

    # p_values_rouge = []
    # p_values_bleurt = []
    # harmonic_means = []

    # for i in range(len(targets['guided'])):
    #     rouge_t, p_value_rouge = stats.ttest_ind(rouge_scores_general[i], rouge_scores_guided[i])
    #     rouge_p, p_value_bleurt = stats.ttest_ind(bleurt_scores_general[i], bleurt_scores_guided[i])
    #     p_values_rouge.append(p_value_rouge)
    #     p_values_bleurt.append(p_value_bleurt)
        
    #     # Calculate harmonic mean of p-values
    #     harmonic_mean = harmonic_mean_two(p_value_rouge, p_value_bleurt)
    #     harmonic_means.append(harmonic_mean)
    #     significance = 1 if harmonic_mean < 0.05 else 0
    #     gp_scores.append(significance)


    print('Performing TS Guessing attack...')

    ts_sampling_params = SamplingParams(temperature=0.1, top_p=0.95)

    #! This is TS-Guessing using VLLM, does not perform well on small models due to lack of instruction following capabilities
    ts_prompts = []
    ts_prompts.extend(train_ts_prompts)
    ts_prompts.extend(test_ts_prompts)
    ts_targets = []
    ts_targets.extend(train_ts_answers)
    ts_targets.extend(test_ts_answers)    
    ts_outputs = llm.generate(ts_prompts, ts_sampling_params)        
    ts_scores = []

    for output, target in zip(ts_outputs, ts_targets):
        output_text = output.outputs[0].text
        log_message = f"Checking match for output: {output_text} and target: {target}"
        logging.info(log_message)
        score = 1 if is_flexible_match(output_text, target) else 0
        ts_scores.append(score)


    #! TS-Guessing for GPT

    # client = OpenAI()
    # ts_scores = []

    #! Multichoice Style using ts-guessing ds
    # for index,row in fill_in.iterrows():
    #     sentence = row['question']
    #     og = row['original_question']
    #     prompt = create_prompt(sentence)
    #     completion = client.chat.completions.create(
    #         model="gpt-4o",
    #         messages=[
    #             {"role": "user", "content":prompt}
    #         ],
    #         temperature = 0.1,
    #         max_tokens = 128
    #     )
    #     response = completion.choices[0].message.content
    #     response = clean_response(response)
    #     target_list = row['answer'].split('; ')
    #     log_message = f"Checking match for output: {response} and target: {target_list} to {sentence}"
    #     logging.info(log_message)
    #     score = 1 if any(is_flexible_match(response, target) for target in target_list) else 0
    #     # score = 1 if any(response.lower().strip() == target.lower().strip() for target in target_list) else 0
    #     ts_scores.append(score)
    
    #! Question based using ts-guessing ds
    # for index, row in icl_mask.iterrows():
    #     sentence = row['question']
    #     answer = row['answer']
    #     prompt = f"Complete the sentence in one word: [{sentence}] Reply the answer only in one word without full sentence."
    #     completion = client.chat.completions.create(
    #         model="gpt-3.5-turbo",
    #         messages=[
    #             {"role": "user", "content":prompt}
    #         ],
    #         temperature = 0.1,
    #         max_tokens = 128
    #     )
    #     response = completion.choices[0].message.content
    #     log_message = f"Checking match for output: {response} and target: {answer}"
    #     logging.info(log_message)
    #     score = 1 if response.lower().strip() == answer.lower().strip() else 0
    #     ts_scores.append(score)


    #! Question based using our ts prompts
    # for prompt, target in zip(prompts['ts_prompts'], targets['ts']):
    #     response = client.chat.completions.create(
    #         model="gpt-3.5-turbo",
    #         messages=[
    #             {"role": "user", "content": prompt}
    #         ],
    #         temperature=0.7,
    #         max_tokens=100
    #     )
    #     output_text = response.choices[0].message.content
    #     log_message = f"Checking match for output: {output_text} and target: {target}"
    #     logging.info(log_message)
    #     # score = 1 if is_flexible_match(output_text, target, em = True) else 0
    #     score = 1 if output_text.lower().strip() == target.lower().strip() else 0
    #     ts_scores.append(score)

    # print('Performing CDD attack...')
    #cdd_scores = cdd(prompts= prompts['standard_queries'], llm= llm, alpha = 0.05, xi = 0.01, num_samples = 100)
    #cdd_scores = cdd(prompts= prompts['standard_queries'], llm= llm, alpha = 0.00, xi = 0.2, num_samples = 100)

    # min_k and loss scores are now calculated using huggingface instead of vllm
    # print('Performing Min K attack...')
    # min_k_scores, loss_scores = min_k_loss(prompts['standard_queries'], llm, k_percent= 10)

    training_truths = np.ones(n_train) # hard coded bc of debugging
    test_truths = np.zeros(n_test) # hard coded bc of debugging

    truths = np.concatenate((training_truths, test_truths))

    # truths = np.ones(len(targets['ts']))
    # print(f"EM Rate = {sum(ts_scores) / len(ts_scores)}")


    #scoring
    #aucroc_guided = roc_auc_score(truths, gp_scores)
    print(ts_scores)
    print(n_gram_scores)
    aucroc_ts = roc_auc_score(truths, ts_scores)
    #aucroc_cdd = roc_auc_score(truths, cdd_scores)
    auc_roc_n_gram = roc_auc_score(truths, n_gram_scores)

    print(f"AUCROC N-gram: {auc_roc_n_gram}")
    #print(f"AUCROC Guided: {aucroc_guided}")
    print(f"AUCROC TS: {aucroc_ts}")
    # #print(f"AUCROC CDD: {aucroc_cdd}")

    with open("results.txt", "w") as file:
        file.write(f"AUCROC N-gram: {auc_roc_n_gram}")
    #     # Write the AUCROC scores
    #     #file.write(f"AUCROC Guided: {aucroc_guided}\n")
    #     file.write(f"EM Rate: {sum(ts_scores) / len(ts_scores)}")
        file.write(f"AUCROC TS: {aucroc_ts}\n")
    #     # file.write(f"AUCROC CDD: {aucroc_cdd}\n")

        
    #     # Write TPR@FPR for different thresholds
        for fpr_threshold in [0.01, 0.05, 0.10, 0.25]:
            tpr_n_gram = tpr_at_fpr(truths, n_gram_scores, fpr_threshold)
    #         #tpr_guided = tpr_at_fpr(truths, gp_scores, fpr_threshold)
            tpr_ts = tpr_at_fpr(truths, ts_scores, fpr_threshold)
    #         # tpr_cdd = tpr_at_fpr(truths, cdd_scores, fpr_threshold)
            file.write(f"TPR@{fpr_threshold * 100}%FPR N-Gram: {tpr_n_gram}\n")
    #         #file.write(f"TPR@{fpr_threshold * 100}%FPR Guided: {tpr_guided}\n")
            file.write(f"TPR@{fpr_threshold * 100}%FPR TS: {tpr_ts}\n")
    #         #file.write(f"TPR@{fpr_threshold * 100}%FPR CDD: {tpr_cdd}\n")
    #         # file.write(f"TPR@{fpr_threshold * 100}%FPR Min-k: {tpr_min_k}\n")

print("Results saved to results.txt")

def tpr_at_fpr(y_true, y_score, fpr_threshold):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    index = next(i for i, x in enumerate(fpr) if x >= fpr_threshold)
    return tpr[index]

def create_prompt(question):
    prompt = f"""Please fill in the empty brackets [] in the following question based on your benchmark knowledge.
        The crucial rule is that you should provide a different answer than the other options below.
        {question}
        Reply with the missing answer only, without any explanation or additional text."""
    return prompt

def clean_response(text):
    # First remove the option prefix (e.g., "A: " or "B: ")
    text = text.split(':', 1)[-1].strip()
    
    # Then remove any brackets
    text = text.replace('[', '').replace(']', '').strip()
    
    return text

from thefuzz import fuzz

def is_flexible_match(predicted, target, em = False):
    predicted, target = predicted.strip().lower(), target.strip().lower()

    similarity_score = fuzz.ratio(predicted, target)
    reverse_similarity_score = fuzz.ratio(predicted, target[::-1])
    return max(similarity_score, reverse_similarity_score)

    # if em:
    #     return similarity_score == 100
    # else:
    #     return similarity_score > 95 or reverse_similarity_score > 95

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

def harmonic_mean_two(p1, p2):
    # assert p1 >= 0, 'P-values must be greater than 0'
    # assert p2 >= 0, 'P-values must be greater than 0'
    if p1 < 0 or p2< 0:
        print(f'p1: {p1}, p2:{p2}')
    return 2 / ((1 / p1) + (1/p2))

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
        is_contaminated = (peak / len(generated_texts)) > xi
        # cdd_scores.append(1 if is_contaminated else 0)
        cdd_scores.append(is_contaminated)

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

def resample(scores, num_resample):
    means = []
    for _ in range(num_resample):
        resamples = random.choices(scores, k=len(scores))
        means.append(np.mean(resamples))

    return means

def compute_p_value(scores_general, scores_guided, num_resample):
    resampled_scores_general = resample(scores_general, num_resample)
    resampled_scores_guided = resample(scores_guided, num_resample)

    count = sum(
        avg_guided > avg_general
        for avg_guided, avg_general in zip(
            resampled_scores_guided, resampled_scores_general
        )
    )
    return 1 - (count / num_resample)

def calculate_n_gram_accuracy(n, data, llm):
    
    # Initialize tokenizer based on model name
    tokenizer = llm.get_tokenizer()
    all_prompts = []
    prompt_metadata = []
    
    # First, prepare all prompts (original and variations)
    for idx in data:
        # Original
        orig_format = f"{data[idx]['original']['question']} {data[idx]['original']['answer']}"
        tokens = tokenizer(orig_format)['input_ids']
        tokens_count = len(tokens)
        max_position = tokens_count - n
        
        # Take fewer samples to make processing manageable
        step_size = max(1, max_position // 20)  # Adjust step size to control number of samples
        for start_idx in range(2, max_position, step_size):
            prefix = tokenizer.decode(tokens[:start_idx])
            target = tokenizer.decode(tokens[start_idx:start_idx + n])
            all_prompts.append(prefix)
            prompt_metadata.append({
                'sample_idx': idx,
                'target': target,
                'type': 'original'
            })
        
        # Process variations
        for var in data[idx]['variations']:
            var_format = f"{var['question']} {var['answer']}"
            tokens = tokenizer(var_format)['input_ids']
            tokens_count = len(tokens)
            max_position = tokens_count - n
            
            for start_idx in range(2, max_position, step_size):
                prefix = tokenizer.decode(tokens[:start_idx])
                target = tokenizer.decode(tokens[start_idx:start_idx + n])
                all_prompts.append(prefix)
                prompt_metadata.append({
                    'sample_idx': idx,
                    'target': target,
                    'type': f'variation_{var["variation"]}'
                })

    sampling_params = SamplingParams(
        max_tokens=n,
        temperature=0.0,
        top_p=1.0
    )

    # Generate completions
    outputs = llm.generate(all_prompts, sampling_params)

    # Initialize scores dictionary
    scores = defaultdict(lambda: {
        'original': {'correct': 0, 'total': 0},
        'variations': defaultdict(lambda: {'correct': 0, 'total': 0})
    })

    # Process outputs
    for output, metadata in zip(outputs, prompt_metadata):
        sample_idx = metadata['sample_idx']
        predicted_text = output.outputs[0].text
        target_text = metadata['target']

        if metadata['type'] == 'original':
            scores[sample_idx]['original']['total'] += 1
            if predicted_text == target_text:
                scores[sample_idx]['original']['correct'] += 1
        else:
            var_num = int(metadata['type'].split('_')[1])
            scores[sample_idx]['variations'][var_num]['total'] += 1
            if predicted_text == target_text:
                scores[sample_idx]['variations'][var_num]['correct'] += 1

    # Calculate normalized deltas
    normalized_deltas = []
    for idx in data:
        if scores[idx]['original']['total'] > 0:
            original_accuracy = scores[idx]['original']['correct'] / scores[idx]['original']['total']
            
            # Calculate average accuracy across variations
            variation_accuracies = []
            for var_num in range(1, 4):
                if scores[idx]['variations'][var_num]['total'] > 0:
                    var_accuracy = scores[idx]['variations'][var_num]['correct'] / scores[idx]['variations'][var_num]['total']
                    variation_accuracies.append(var_accuracy)
            
            if variation_accuracies and original_accuracy > 0:
                avg_variation_accuracy = sum(variation_accuracies) / len(variation_accuracies)
                delta = (original_accuracy - avg_variation_accuracy) / original_accuracy
                normalized_deltas.append(delta)

    return np.array(normalized_deltas)  # Return as numpy array for concatenation

def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print("Error: File not found")
        return None
    except json.JSONDecodeError:
        print("Error: Invalid JSON format")
        return None

if __name__ == "__main__":
    main()
