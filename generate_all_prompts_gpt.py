import asyncio
import json
import nltk
import os
import string
import difflib
from datetime import datetime
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from tqdm import tqdm
from openai import OpenAI
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset
import argparse
import random

class DatasetConfig:
    """Configuration for different datasets' key mappings"""
    
    CONFIGS = {
        'question_based': {
            "metamath": {
                "question_key": "query",
                "answer_key": "response"
            },
            "gsm8k": {
                "question_key": "question",
                "answer_key": "answer"
            },
            "truthful_qa": {
                "subset": "generation",
                "question_key": "question",
                "answer_key": "best_answer"
            }
        },
        'multiple_choice': {
            "metamath": {
                "question_key": "query",
                "answer_key": "response"
            },
            "gsm8k": {
                "question_key": "question",
                "answer_key": "answer"
            },
            "truthful_qa": {
                "subset": "multiple_choice",
                "question_key": "question",
                "answer_key": "mc1_targets",
                "choices_key": "choices",
                "labels_key": "labels"
            },
            "cais/mmlu": {
                "subset": "all",
                "split": "auxiliary_train",
                "question_key": "question",
                "answer_key": "choices",
                "choices_key": "choices",
                "correct_answer_key": "answer"
            },
            "austinrbrown/contamination-mmlu": {
                "split": "contaminated", # this is the one the model has seen, "clean is the other option"
                "question_key": "question",
                "answer_key": "choices",
                "choices_key": "choices",
                "correct_answer_key": "answer"
            }
        }
    }

    @staticmethod
    def get_keys(dataset_name: str, question_type: str) -> Dict[str, str]:
        """Get the question and answer keys for a given dataset"""
        dataset_name = dataset_name.lower()
        if dataset_name not in DatasetConfig.CONFIGS[question_type]:
            raise ValueError(f"Unknown dataset: {dataset_name}, for {question_type}")
        return DatasetConfig.CONFIGS[question_type][dataset_name]

@dataclass
class ErrorLog:
    def __init__(self, log_file: str = 'error_log.json'):
        self.log_file = log_file
        self._ensure_log_file()
    
    def _ensure_log_file(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                json.dump([], f)
    
    def log_error(self, error_type: str, details: Dict[str, Any]):
        entry = {
            'timestamp': datetime.now().isoformat(),
            'type': error_type,
            'details': details
        }
        
        with open(self.log_file, 'r+') as f:
            data = json.load(f)
            data.append(entry)
            f.seek(0)
            json.dump(data, f, indent=4)
            f.truncate()

class Prompt:
    def __init__(self) -> None:
        self.prompts = {
            "general": (
                "INSTRUCTION:\n"
                "You are provided with the FIRST PIECE of a question from the train split of a dataset.\n"
                "Finish the SECOND PIECE of the question as exactly appeared in the dataset.\n"
                "Only rely on the original form of the question in the dataset to finish the SECOND PIECE.\n\n"
                "FIRST PIECE:\n"
                "{first_piece}\n\n"
                "SECOND PIECE:\n"
            ),
            "guided": (
                "Instruction:\n"
                "You are provided with the FIRST PIECE of a question from the train split of the {dataset_name} dataset.\n"
                "Finish the SECOND PIECE of the question as exactly appeared in the dataset.\n"
                "Only rely on the original form of the question in the dataset to finish the SECOND PIECE.\n\n"
                "FIRST PIECE:\n"
                "{first_piece}\n\n"
                "SECOND PIECE:\n"
            ),
            "rewrite": (
                "Please act as a general content rewriter to paraphrase the question and answer presented below.\n"
                "Please follow these instructions:\n"
                "1. Paraphrase the question by rewording it with new expressions and sentence structures.\n"
                "2. Do not change the meaning of the question or the answer.\n"
                "3. Stay as close to the original content and style as possible.\n"
                "4. Format your response as follows:\n"
                "The rewritten question: <your rewritten question>\n"
                "The rewritten answer: <your rewritten answer>"
            )
        }

    def get_prompt(self, prompt_type: str) -> str:
        return self.prompts.get(prompt_type, "Invalid prompt type")

class DatasetProcessor:
    def __init__(self, dataset_name: str, openai_api_key: str, question_type:str):
        self.dataset_name = dataset_name
        self.client = OpenAI(api_key=openai_api_key)
        self.error_logger = ErrorLog()
        self.prompt = Prompt()
        self.tagger = nltk.pos_tag
        self.config = DatasetConfig.get_keys(dataset_name, question_type)
        self.question_type = question_type
        self.question_key = self.config["question_key"]
        self.answer_key = self.config["answer_key"]
        self.subset = self.config.get('subset', None)

        if question_type == 'multiple_choice':
            if dataset_name == 'truthful_qa':
                self.choices_key = "choices"
                self.labels_key = "labels"
            elif dataset_name == 'cais/mmlu':
                self.choices_key = "choices"
                self.correct_answer_key = "answer"
            elif dataset_name == 'austinrbrown/contamination-mmlu':
                self.choices_key = "choices"
                self.correct_answer_key = "answer"

        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        
    async def process_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single example through all components."""
        try:
            result = {
                'original': {
                    'question': example[self.question_key],
                    'answer': example[self.answer_key]
                }
            }
            
            # Generate guided prompts
            splits = self._guided_prompt_split(example)
            result['guided'] = {
                'general_prompt': self.prompt.get_prompt("general").format(first_piece=splits['guided_prompt_part_1']),
                'guided_prompt': self.prompt.get_prompt("guided").format(
                    dataset_name=self.dataset_name,
                    first_piece=splits['guided_prompt_part_1']
                ),
                'answer': splits['guided_prompt_part_2']
            }
            
            # Generate text substitution
            ts_prompt, mask = await self._generate_ts_prompt(example)
            result['ts'] = {
                'masked': ts_prompt,
                'mask': mask
            }
            
            # Generate variations using GPT-3.5
            variations = await self._generate_variations(example)
            # Get correct answer for multiple choice questions
            if hasattr(self, 'choices_key'):
                choices = example[self.choices_key]
                if self.dataset_name == 'truthful_qa':
                    labels = example[self.labels_key]
                    correct_idx = labels.index(1)
                else:  # mmlu
                    correct_idx = example[self.correct_answer_key]
                correct_answer = choices[correct_idx]
            else:
                correct_answer = example[self.answer_key]
            result['n_gram'] = {
                'original': {
                    'question': example[self.question_key],
                    'answer': correct_answer
                },
                'variations': variations
            }
            
            return result
            
        except Exception as e:
            self.error_logger.log_error('processing_error', {
                'example': example,
                'error': str(e)
            })
            return None

    def _guided_prompt_split(self, example: Dict[str, Any]) -> Dict[str, str]:
        """Split text for guided prompts."""
        text = example[self.question_key]
        sentences = nltk.sent_tokenize(text)
        
        if len(sentences) == 1:
            mid_point = len(text) // 2
            return {
                'guided_prompt_part_1': text[:mid_point],
                'guided_prompt_part_2': text[mid_point:]
            }
        else:
            first_part_length = len(sentences) // 2
            return {
                'guided_prompt_part_1': ' '.join(sentences[:first_part_length]),
                'guided_prompt_part_2': ' '.join(sentences[first_part_length:])
            }
    async def _generate_ts_prompt(self, example: Dict[str, Any]) -> Tuple[str, str]:
        """Router function to appropriate ts_prompt generator based on question type"""
        if self.question_type == 'multiple_choice':
            return await self._generate_multiple_choice_ts_prompt(example)
        return await self._generate_question_based_ts_prompt(example)

    async def _generate_multiple_choice_ts_prompt(self, example: Dict[str, Any]) -> Tuple[str, str]:
        """Generate text substitution prompt for multiple choice questions"""
        try:
            choices = example[self.choices_key]
            
            # Get correct answer index
            if self.dataset_name == 'truthful_qa':
                labels = example[self.labels_key]
                correct_idx = labels.index(1)
            else:  # mmlu
                correct_idx = example[self.correct_answer_key]
            
            # Select random incorrect answer to mask
            incorrect_indices = [i for i in range(len(choices)) if i != correct_idx]
            mask_idx = random.choice(incorrect_indices)
            masked_answer = choices[mask_idx]
            
            # Create masked choices
            masked_choices = choices.copy()
            masked_choices[mask_idx] = "[MASK]"
            
            final_prompt = (
                f"Question: {example[self.question_key]}\n"
                f"Choices: {', '.join(masked_choices)}"
            )
            
            return final_prompt, masked_answer
            
        except Exception as e:
            self.error_logger.log_error('ts_prompt_generation_error', {
                'error': str(e),
                'example': example
            })
            return '', ''

    async def _generate_question_based_ts_prompt(self, example: Dict[str, Any]) -> Tuple[str, str]:
        """Generate text substitution prompt with improved word selection and masking."""
        try:
            # dataset specific processing
            if self.dataset_name == 'truthful_qa':
                if len(example[self.question_key]) <= 4:
                    #skip examples that are too short
                    return '', ''
                if 'Indexical Error' in example['category']:
                    #skip indexical error
                    return '', ''
            if self.dataset_name == 'cais/mmlu':
                raise ValueError('MMLU not supported for Question Based TS-Guessing attack, use Multiple Choice instead')
            else:
                #add general processing
                pass

            text = example[self.question_key]
            # Get significant words through POS tagging
            tokens = text.split()
            tags = self.tagger(tokens, tagset='universal')
            significant_words = [
                word for word, tag in tags 
                if tag in ['NOUN', 'ADJ', 'VERB'] 
                and len(word) > 2  # Skip very short words
            ]
            
            if not significant_words:
                return "No significant words found.", ""

            # Create prompt for word selection
            word_selection_prompt = (
                # "Select the most mathematically significant word from these options:\n"
                # f"Words: {', '.join(significant_words)}\n"
                # f"Text: {text}\n"
                # "Consider:\n"
                # "- Nouns representing quantities or mathematical objects\n"
                # "- Verbs representing mathematical operations\n"
                # "- Adjectives describing mathematical properties\n"
                # "Reply with only the selected word."
                "Select the most contextually significant word from these options:\n"
                f"Words: {', '.join(significant_words)}\n"
                f"Text: {text}\n"
                "Consider:\n"
                "- Nouns representing key concepts or entities\n"
                "- Verbs representing critical actions or processes\n"
                "- Adjectives describing important qualities or properties\n"
                "Reply with only the selected word."
            )

            # Get GPT's word selection
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": word_selection_prompt}],
                max_tokens=5,
                temperature=0.3
            )

            selected_word = response.choices[0].message.content.strip()

            # Verify and potentially correct the word
            if selected_word not in significant_words:
                # Try to find closest match
                closest = difflib.get_close_matches(
                    selected_word.lower(),
                    [word.lower() for word in significant_words],
                    n=1,
                    cutoff=0.8
                )
                
                if closest:
                    # Find original word with matching lowercase version
                    selected_word = next(
                        word for word in significant_words 
                        if word.lower() == closest[0]
                    )
                else:
                    # Log unexpected word selection
                    self.error_logger.log_error('unexpected_word_selection', {
                        'text': text,
                        'selected': selected_word,
                        'candidates': significant_words
                    })

            # Create masked version of text
            word_lower = selected_word.lower()
            masked_text = text

            if word_lower not in text.lower():
                self.error_logger.log_error('masking_error', {
                    'text': text,
                    'selected_word': selected_word
                })
                return "Masking failed.", ""
            
            # Find and replace the word, preserving case
            for i in range(len(text) - len(selected_word) + 1):
                if text[i:i + len(selected_word)].lower() == word_lower:
                    masked_text = (
                        text[:i] + 
                        "[MASK]" + 
                        text[i + len(selected_word):]
                    )
                    break

            # Create final prompt
            final_prompt = (
                "Complete the sentence in one word:\n\n"
                f"{masked_text}\n"
                "Reply with only the missing word."
            )

            return final_prompt, selected_word

        except Exception as e:
            self.error_logger.log_error('ts_prompt_generation_error', {
                'error': str(e),
                'text': example.get(self.question_key, 'unknown')
            })
            return '', ''

    async def _generate_variations(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate variations using GPT-3.5 in parallel."""
        async def generate_single_variation(variation_number: int) -> Dict[str, Any]:
            try:
                # Get the correct answer based on dataset type
                if hasattr(self, 'choices_key'):
                    choices = example[self.choices_key]
                    if self.dataset_name == 'truthful_qa':
                        labels = example[self.labels_key]
                        correct_idx = labels.index(1)
                        correct_answer = choices[correct_idx]
                    else:  # mmlu
                        correct_idx = example[self.correct_answer_key]
                        correct_answer = choices[correct_idx]
                else:
                    correct_answer = example[self.answer_key]

                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model="gpt-3.5-turbo-0125",
                    messages=[
                        {"role": "system", "content": self.prompt.get_prompt("rewrite")},
                        {"role": "user", "content": f"Question: {example[self.question_key]}\nAnswer: {correct_answer}"}
                    ],
                    temperature=0.7,
                    top_p=0.9
                )
                
                content = response.choices[0].message.content
                try:
                    question_part = content.split("The rewritten question: ")[1].split("The rewritten answer: ")[0].strip()
                    answer_part = content.split("The rewritten answer: ")[1].strip()
                    
                    return {
                        'question': question_part,
                        'answer': answer_part,
                        'variation': variation_number
                    }
                except IndexError as e:
                    self.error_logger.log_error('response_parsing_error', {
                        'variation_number': variation_number,
                        'content': content,
                        'error': str(e)
                    })
                    return None
            
            except Exception as e:
                self.error_logger.log_error('variation_generation_error', {
                    'variation_number': variation_number,
                    'error': str(e)
                })
                return None

        variations = await asyncio.gather(
            *[generate_single_variation(i) for i in range(1, 4)]
        )
        
        return [v for v in variations if v is not None]

    async def process_dataset(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process entire dataset with progress bar."""
        results = []
        with tqdm(total=len(dataset), desc="Processing dataset") as pbar:
            for example in dataset:
                result = await self.process_example(example)
                if result:
                    results.append(result)
                pbar.update(1)
        return results

def load_data(input_file: str | None, download: bool, dataset_name: str, question_type: str, num_lines: int):
    config = DatasetConfig.get_keys(dataset_name, question_type)
    
    if not download:
        dataset = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                dataset.append(json.loads(line))
        return dataset
    else:
        # Stream from Hugging Face dataset
        subset = config.get('subset', None)
        split = config.get('split', "train")
        if subset:
            dataset = load_dataset(dataset_name, subset, split=split, streaming=True)
        else:
            dataset = load_dataset(dataset_name, split=split, streaming=True)
        return list(dataset.take(num_lines))


async def main():
    parser = argparse.ArgumentParser(description='Generate prompts for dataset processing')
    
    # Required arguments
    parser.add_argument('--dataset_name', 
                       type=str, 
                       required=True,
                       choices=['metamath', 'gsm8k', 'truthful_qa', 'cais/mmlu', 'austinrbrown/contamination-mmlu'],
                       help='Name of the dataset to process')
    
    parser.add_argument('--openai_key', 
                       type=str, 
                       required=True,
                       help='OpenAI API key')
    
    parser.add_argument('--question_type',
                        type=str,
                        required=True,
                        choices=['question_based', 'multiple_choice'],
                        help='For TS-Guessing, Generation or Multiple Choice Based')
    
    # Optional arguments

    parser.add_argument('--input_file', 
                    type=str,
                    default = None, 
                    help='Path to input dataset file')

    parser.add_argument('--output_file', 
                       type=str, 
                       default='processed_dataset.json',
                       help='Path to save processed dataset (default: processed_dataset.json)')
    
    parser.add_argument('--error_log', 
                       type=str, 
                       default='error_log.json',
                       help='Path to error log file (default: error_log.json)')

    parser.add_argument('--num_lines', type=int, default=1000,
                       help='Number of lines to stream when downloading')

    parser.add_argument('--download', type=bool, default=False,
                       help='Whether to download from HF instead of local file')

    args = parser.parse_args()
    
    
    dataset = []

    dataset = load_data(
        args.input_file, 
        args.download, 
        args.dataset_name,
        args.question_type,
        args.num_lines
    )

    processor = DatasetProcessor(
        dataset_name=args.dataset_name,
        openai_api_key=args.openai_key,
        question_type=args.question_type, 
    )

    results = await processor.process_dataset(dataset)

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    asyncio.run(main())