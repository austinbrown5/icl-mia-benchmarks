from vllm import LLM, SamplingParams
from datasets import load_dataset
import nltk
from nltk import word_tokenize
from nltk.tag import StanfordPOSTagger
import random
import numpy as np
from prompts import Prompt
from sklearn.metrics import roc_auc_score, roc_curve
from scipy import stats
from rouge_score import rouge_scorer

def main():
    # Usage
    model_name = "meta-math/MetaMath-Mistral-7B"
    dataset_name = "meta-math/MetaMathQA"

    # load data
    meta_math_ds = load_dataset(dataset_name, streaming = True) # load subset
    meta_math_ds = meta_math_ds.take(1000) # take only the first 1000 examples while debugging

    llm = LLM(model=model_name, dtype="half")
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95)

    meta_math_training = meta_math_ds['train']

    # loop through the data and generate our different prompts 
    prompts = {
        "general_prompts": [],
        "guided_prompts": [],
        "ts_prompts": [],
        "standard_queries": meta_math_training['query']
    }

    targets = {
        "guided": [],
        "ts": [],
        "answers": meta_math_training['response']
    }
    
    # load stanford pos tagger, need to change to not hard code paths
    os.environ['CLASSPATH']="/usr/project/xtmp/arb153/icl-mia-benchmarks/stanford-postagger-full-2020-11-17/stanford-postagger.jar"
    os.environ["STANFORD_MODELS"] = "/usr/project/xtmp/arb153/icl-mia-benchmarks/stanford-postagger-full-2020-11-17/models"
    tagger = StanfordPOSTagger('english-bidirectional-distsim.tagger')

    #going to go through the dataset and generate our unique prompts, do not need these for our min_k, perplexity, or cdd attacks
    for data_point in meta_math_training:

        #split for guided prompting
        splits = guided_prompt_split_fn(data_point, 'query')

        guided_prompt_insert = splits['guided_prompt_part_1']
        guided_prompt_target = splits['guided_prompt_part_2']

        prompt = Prompt()
        guided_prompt = prompt.get_prompt("guided").format(dataset_name = dataset_name,
                                                           first_piece = guided_prompt_insert)
        general_prompt = prompt.get_prompt("general").format(first_piece = guided_prompt_insert)

        #ts guessing prompt and target generation, question based method
        ts_prompt, ts_target = ts_guessing_prompt(data_point, tagger, 'query')

        prompts['general_prompts'].append(general_prompt)
        prompts['guided_prompts'].append(guided_prompt)
        

        prompts['ts_prompts'].append(ts_prompt)

        targets['guided'].append(guided_prompt_target)
        targets['ts'].append(ts_target)
    
    general_outputs = llm.generate(prompts['general_promts'], sampling_params)
    guided_outputs = llm.generate(prompts['guided_prompts'], sampling_params)

    ts_outputs = llm.generate(prompts['ts_prompts'], sampling_params)

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    general_scores = [scorer.score(ref, hyp)['rougeL'].fmeasure for ref, hyp in zip(targets['answers'], [output.outputs[0].text for output in general_outputs])]
    guided_scores = [scorer.score(ref, hyp)['rougeL'].fmeasure for ref, hyp in zip(targets['guided'], [output.outputs[0].text for output in guided_outputs])]

    t_statistic, p_value = stats.ttest_rel(guided_scores, general_scores)

    guided_scores = [1 if p_value < 0.05 else 0 for _ in range(len(guided_scores))] # change to use p-value as membership score ( try p value, 1 - p value)

    ts_scores = [1 if output.outputs[0].text.strip().lower() == target.strip().lower() else 0 
                 for output, target in zip(ts_outputs, targets['ts'])]

    cdd_scores = cdd(prompts= prompts['standard_queries'], llm= llm, alpha = 0.05, xi = 0.01 ) #this will be binary 1s and 0s
    min_k_scores, loss_scores = min_k_loss(prompts['standard_queries'], llm, k_percent= 10)

    truths = np.ones(len(prompts)) # we know the model has been contaminated

    assert len(truths) == len(cdd_scores)
    assert len(truths) == len(min_k_scores)
    assert len(truths) == len(loss_scores)
    assert len(truths) == len(guided_scores)
    assert len(truths) == len(ts_scores)

    #scoring
    aucroc_guided = roc_auc_score(truths, guided_scores)
    aucroc_ts = roc_auc_score(truths, ts_scores)
    aucroc_cdd = roc_auc_score(truths, cdd_scores)
    aucroc_min_k = roc_auc_score(truths, min_k_scores)

    print(f"AUCROC Guided: {aucroc_guided}")
    print(f"AUCROC TS: {aucroc_ts}")
    print(f"AUCROC CDD: {aucroc_cdd}")
    print(f"AUCROC Min-k: {aucroc_min_k}")

    for fpr_threshold in [0.01, 0.05, 0.10, 0.25]:
        tpr_guided = tpr_at_fpr(truths, guided_scores, fpr_threshold)
        tpr_ts = tpr_at_fpr(truths, ts_scores, fpr_threshold)
        tpr_cdd = tpr_at_fpr(truths, cdd_scores, fpr_threshold)
        tpr_min_k = tpr_at_fpr(truths, min_k_scores, fpr_threshold)

        print(f"TPR@1%FPR Guided: {tpr_guided}")
        print(f"TPR@1%FPR TS: {tpr_ts}")
        print(f"TPR@1%FPR CDD: {tpr_cdd}")
        print(f"TPR@1%FPR Min-k: {tpr_min_k}")

def tpr_at_fpr(y_true, y_score, fpr_threshold):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    index = next(i for i, x in enumerate(fpr) if x >= fpr_threshold)
    return tpr[index]

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
    
def get_ll_vllm(sentence, llm):
    # we can only get the logprobs for up to the top 100 tokens with vllm
    sampling_params = SamplingParams(temperature=0.0, max_tokens=0, logprobs=100)
    outputs = llm.generate_with_logprobs([sentence], sampling_params)
    logprobs = outputs[0].outputs[0].logprobs

    all_prob = []
    for token_logprobs in logprobs:
        token_prob = np.exp(token_logprobs['token_logprobs'])
        all_prob.append(token_logprobs['token_logprobs'])
    
    ll = sum(all_prob)  # log-likelihood
    ppl = np.exp(-ll / len(all_prob))  # perplexity
    prob = np.exp(ll)  # probability
    
    return prob, ll, ppl, all_prob

def min_k_loss(texts, llm, k_percent=10):
    # these are a different type of generation that the others
    sampling_params = SamplingParams(temperature=0.0, max_tokens=0, logprobs=100)
    outputs = llm.generate_with_logprobs(texts, sampling_params)

    min_ks = []
    losses = []
    for output in outputs:
        token_logprobs = [lp['token_logprobs'] for lp in output.outputs[0].logprobs]
        logprobs = np.array(token_logprobs)

        k = max(1, int(len(logprobs) * k_percent / 100))
        topk = np.sort(logprobs)[:k]
        min_k_score = np.mean(topk).item()

        min_ks.append(min_k_score)

        loss = -np.sum(logprobs)
        losses.append(loss)


    return min_ks, losses

def cdd(prompts, llm, alpha = 0.05, xi = 0.01):
    sampling_params_multiple = SamplingParams(
        temperature=0.8,
        n=50
    )
    sampling_params_greedy = SamplingParams(
        temperature=0,
        n=1
    )
    cdd_scores = []
    for prompt in prompts:
        samples = llm.generate([prompt] * 50, sampling_params_multiple)
        generated_texts = [output.outputs[0].text for output in samples]

        greedy_sample = llm.generate([prompt], sampling_params_greedy)[0]
        s_0 = greedy_sample.text

        peak = get_peak(generated_texts, s_0, alpha)
        # is_contaminated = peak > xi
        # if is_contaminated:
        #     cdd_scores.append(1)
        # else:
        #     cdd_scores.append(0)
        cdd_scores.append(peak) #* we want a membership score instead of a binary value, 
        #? these scores will be very low, is that alright
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

if __name__ == "main":
    main()
