"""
Benchmark Testing Script for GPQA Dataset
==========================================

Modified from Omni-MATH benchmark script to run on GPQA train/test splits with:
- Support for GPQA data format (Question, Correct Answer, Incorrect Answers, Domain, Subdomain)
- Multiple choice question handling with answer shuffling
- Enhanced prompts with self-consistency and self-refine
- OpenAI O3 model for answer evaluation
- MajorityVoteAgent for intelligent answer aggregation
- Checkpointing for resumable runs

Usage:
    python benchmark_gpqa.py --data-file ./data/gpqa_diamond_train.csv
    python benchmark_gpqa.py --data-file ./data/gpqa_diamond_test.csv --start-index 0 --end-index 50
    python benchmark_gpqa.py --resume-from ./checkpoints/checkpoint_20251219_120000.json
"""

import json
import time
import os
import csv
import re
import base64
import requests
import numpy as np
import signal
import sys
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter
from openai import OpenAI
from pathlib import Path
from datetime import datetime
import pandas as pd


@dataclass
class BenchmarkQuestion:
    """Represents a single benchmark question"""
    question_id: str
    question: str
    image: str  # Path to local image file (empty for GPQA)
    answer: str  # The correct answer text
    answer_type: str  # multipleChoice for GPQA
    category: str  # High-level domain (Physics, Chemistry, Biology)
    subject: str  # Subdomain
    author: str
    rationale: str
    difficulty: str = ""
    # GPQA specific fields
    incorrect_answers: List[str] = None  # List of incorrect answer choices
    answer_choices: Dict[str, str] = None  # Shuffled choices: {'A': '...', 'B': '...', ...}
    correct_letter: str = ""  # Letter of correct answer after shuffling


@dataclass
class TestResult:
    """Result for a single question test"""
    question_id: str
    question: str
    correct_answer: str
    predicted_answer: str
    is_correct: bool
    reasoning: str
    answer_type: str
    category: str
    subject: str
    has_image: bool
    evaluation_reasoning: str = ""
    # Self-consistency specific fields
    all_predictions: List[str] = None
    prediction_counts: Dict[str, int] = None
    confidence: float = None
    # Self-refine specific fields
    initial_answer: str = None
    was_refined: bool = False
    # Agent analysis field
    agent_analysis: str = ""
    difficulty: str = ""
    # GPQA specific
    correct_letter: str = ""
    predicted_letter: str = ""
    answer_choices: Dict[str, str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'question_id': self.question_id,
            'question': self.question,
            'correct_answer': self.correct_answer,
            'predicted_answer': self.predicted_answer,
            'is_correct': self.is_correct,
            'reasoning': self.reasoning,
            'answer_type': self.answer_type,
            'category': self.category,
            'subject': self.subject,
            'has_image': self.has_image,
            'evaluation_reasoning': self.evaluation_reasoning,
            'all_predictions': self.all_predictions,
            'prediction_counts': self.prediction_counts,
            'confidence': self.confidence,
            'initial_answer': self.initial_answer,
            'was_refined': self.was_refined,
            'agent_analysis': self.agent_analysis,
            'difficulty': self.difficulty,
            'correct_letter': self.correct_letter,
            'predicted_letter': self.predicted_letter,
            'answer_choices': self.answer_choices
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'TestResult':
        """Create TestResult from dictionary"""
        return cls(**data)


@dataclass
class CheckpointData:
    """Data structure for checkpoint saves"""
    timestamp: str
    data_file: str
    start_index: int
    end_index: Optional[int]
    current_index: int
    total_questions: int
    completed_question_ids: List[str]
    results: List[Dict]
    config: Dict
    elapsed_time: float

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'CheckpointData':
        return cls(**data)


class MajorityVoteAgent:
    """
    Agent that collects predictions and determines the majority choice.
    """

    def __init__(self, client: OpenAI = None, use_llm: bool = True, model: str = "gpt-4o"):
        self.client = client
        self.use_llm = use_llm and client is not None
        self.model = model
        self.predictions = []
        self.question = None
        self.answer_type = None

    def reset(self):
        self.predictions = []
        self.question = None
        self.answer_type = None

    def set_question(self, question: str, answer_type: str):
        self.question = question
        self.answer_type = answer_type

    def add_prediction(self, prediction: str, reasoning: str = ""):
        self.predictions.append({
            'answer': prediction.strip(),
            'reasoning': reasoning
        })

    def add_predictions(self, predictions: List[str]):
        for pred in predictions:
            self.add_prediction(pred)

    def _normalize_answer(self, answer: str) -> str:
        normalized = answer.strip().upper()
        # For multiple choice, extract just the letter
        if len(normalized) >= 1 and normalized[0] in 'ABCD':
            return normalized[0]
        return normalized

    def _simple_majority_vote(self) -> Tuple[str, Dict[str, int], float, str]:
        if not self.predictions:
            return "", {}, 0.0, "No predictions to analyze"

        normalized_map = {}
        counts = Counter()

        for pred in self.predictions:
            answer = pred['answer']
            normalized = self._normalize_answer(answer)

            if normalized not in normalized_map:
                normalized_map[normalized] = answer

            counts[normalized] += 1

        most_common = counts.most_common()
        winner_normalized = most_common[0][0]
        winner_count = most_common[0][1]

        winner = normalized_map[winner_normalized]

        total = len(self.predictions)
        confidence = winner_count / total

        analysis = f"Simple majority vote analysis:\n"
        analysis += f"Total predictions: {total}\n"
        analysis += f"Unique answers: {len(counts)}\n"
        analysis += f"Distribution:\n"
        for ans, count in most_common:
            pct = count / total * 100
            analysis += f"  - '{normalized_map[ans]}': {count} ({pct:.1f}%)\n"
        analysis += f"Winner: '{winner}' with {confidence:.1%} confidence"

        original_counts = {normalized_map[k]: v for k, v in counts.items()}

        return winner, original_counts, confidence, analysis

    def _llm_majority_vote(self) -> Tuple[str, Dict[str, int], float, str]:
        if not self.predictions:
            return "", {}, 0.0, "No predictions to analyze"

        predictions_text = "\n".join([
            f"{i + 1}. {pred['answer']}"
            for i, pred in enumerate(self.predictions)
        ])

        prompt = f"""You are an expert answer aggregation agent for multiple choice questions. Analyze these predictions and determine the majority answer.

Question: {self.question[:500]}...
Answer Type: {self.answer_type}

Predictions collected:
{predictions_text}

Your task:
1. Group equivalent answers together (e.g., "A", "A.", "(A)", "Option A" are the same)
2. Determine which answer has the majority
3. Calculate confidence as (majority count / total predictions)

Respond in this exact format:
GROUPS:
- Group A: predictions that chose A
- Group B: predictions that chose B
- etc.

MAJORITY_ANSWER: [Just the letter A, B, C, or D]
MAJORITY_COUNT: [number of predictions in majority group]
TOTAL_PREDICTIONS: {len(self.predictions)}
CONFIDENCE: [majority_count / total as decimal]

ANALYSIS: [Brief explanation of your decision]"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1000
            )

            result = response.choices[0].message.content.strip()

            majority_match = re.search(r'MAJORITY_ANSWER:\s*([A-D])', result, re.IGNORECASE)
            confidence_match = re.search(r'CONFIDENCE:\s*([\d.]+)', result)
            analysis_match = re.search(r'ANALYSIS:\s*(.+)', result, re.DOTALL)

            winner = majority_match.group(1).upper() if majority_match else ""
            confidence = float(confidence_match.group(1)) if confidence_match else 0.0
            analysis = analysis_match.group(1).strip() if analysis_match else result

            _, simple_counts, _, _ = self._simple_majority_vote()

            return winner, simple_counts, confidence, f"LLM Analysis:\n{analysis}"

        except Exception as e:
            print(f"  ⚠️ LLM analysis failed: {e}, falling back to simple voting")
            return self._simple_majority_vote()

    def get_majority_answer(self) -> Tuple[str, Dict[str, int], float, str]:
        if self.use_llm:
            return self._llm_majority_vote()
        else:
            return self._simple_majority_vote()

    def analyze_disagreement(self) -> Dict:
        if not self.predictions or len(self.predictions) < 2:
            return {"has_disagreement": False}

        unique_answers = set(self._normalize_answer(p['answer']) for p in self.predictions)

        if len(unique_answers) == 1:
            return {
                "has_disagreement": False,
                "message": "All predictions agree"
            }

        return {
            "has_disagreement": True,
            "unique_answers": len(unique_answers),
            "message": f"Found {len(unique_answers)} different answers"
        }


class BaselinePromptsManager:
    """Manages baseline prompts from prompts.json"""

    def __init__(self, prompts_file: str = "prompts.json"):
        self.prompts_file = prompts_file
        self.prompts = {}
        self.model_settings = {}
        self.load_prompts()

    def load_prompts(self):
        if os.path.exists(self.prompts_file):
            with open(self.prompts_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.prompts = data.get('prompts', {})
            self.model_settings = data.get('model_settings', {})
            print(f"  ✓ Loaded {len(self.prompts)} baseline prompts from {self.prompts_file}")
        else:
            # Default prompts for GPQA (multiple choice)
            self.prompts = {
                'zero_shot': {
                    'template': """Answer the following multiple choice question.

Question: {question}

{choices}

Provide only the letter of the correct answer (A, B, C, or D).

Answer:""",
                    'sampling_params': {'temperature': 0, 'max_tokens': 2000}
                },
                'zero_shot_cot': {
                    'template': """Answer the following multiple choice question. Think step by step before giving your final answer.

Question: {question}

{choices}

Think through this carefully, showing your reasoning. At the end, clearly state your final answer as a single letter (A, B, C, or D).

Final Answer:""",
                    'sampling_params': {'temperature': 0, 'max_tokens': 3000}
                },
                'zero_shot_self_consistency': {
                    'template': """Answer the following multiple choice question. Think step by step and explain your reasoning.

Question: {question}

{choices}

Work through this problem carefully. After your reasoning, clearly state your final answer as a single letter (A, B, C, or D).

Final Answer:""",
                    'sampling_params': {'temperature': 0.7, 'num_samples': 5, 'max_tokens': 3000}
                },
                'self_refine': {
                    'template': """Answer the following multiple choice question. After answering, verify your reasoning.

Question: {question}

{choices}

Provide your answer as a single letter (A, B, C, or D).

Answer:""",
                    'sampling_params': {'temperature': 0, 'max_tokens': 3000}
                },
                'expert': {
                    'template': """You are an expert scientist with deep knowledge in physics, chemistry, and biology. Answer the following graduate-level multiple choice question with careful reasoning.

Question: {question}

{choices}

Approach this systematically:
1. Identify the key concepts and domain knowledge required
2. Analyze each answer choice
3. Eliminate incorrect options with clear reasoning
4. Select the best answer

Provide your final answer as a single letter (A, B, C, or D).

Final Answer:""",
                    'sampling_params': {'temperature': 0, 'max_tokens': 4000}
                }
            }
            print(f"  ✓ Using default GPQA prompts (no {self.prompts_file} found)")

    def get_prompt_template(self, prompt_key: str) -> Optional[str]:
        if prompt_key in self.prompts:
            return self.prompts[prompt_key].get('template', '')
        return None

    def get_sampling_params(self, prompt_key: str) -> Dict:
        if prompt_key in self.prompts:
            return self.prompts[prompt_key].get('sampling_params', {})
        return {}

    def is_self_consistency_prompt(self, prompt_key: str) -> bool:
        if 'self_consistency' in prompt_key.lower():
            return True
        params = self.get_sampling_params(prompt_key)
        if params.get('num_samples', 1) > 1:
            return True
        return False

    def is_self_refine_prompt(self, prompt_key: str) -> bool:
        if 'self_refine' in prompt_key.lower():
            return True
        return False

    def list_available_prompts(self) -> List[str]:
        return list(self.prompts.keys())


class BenchmarkPromptManager:
    """Manages loading enhanced prompts for different question types"""

    def __init__(self, enhancement_base_dir: str, use_enhanced: bool = True,
                 enhancement_type: str = 'specific', target_strategy: str = 'zero_shot'):
        self.enhancement_base_dir = Path(enhancement_base_dir) if enhancement_base_dir else None
        self.use_enhanced = use_enhanced
        self.enhancement_type = enhancement_type
        self.target_strategy = target_strategy
        self.category_prompts = {}

        if use_enhanced and self.enhancement_base_dir:
            self.load_all_category_prompts()

    def normalize_name(self, name: str) -> str:
        return name.lower().replace('_', ' ').replace('-', ' ').strip()

    def load_all_category_prompts(self):
        print(f"\n  Loading enhanced prompts from: {self.enhancement_base_dir}")
        print(f"  Enhancement type: {self.enhancement_type}")
        print(f"  Target strategy: {self.target_strategy}")

        if not self.enhancement_base_dir.exists():
            print(f"    ⚠️  Enhancement directory not found")
            return

        loaded_count = 0

        for item_dir in self.enhancement_base_dir.iterdir():
            if not item_dir.is_dir():
                continue

            dir_name = item_dir.name
            strategy = None
            category = None

            for prefix in ['zero_shot_cot_', 'zero_shot_', 'self_refine_']:
                if dir_name.startswith(prefix):
                    strategy = prefix.rstrip('_')
                    category = dir_name[len(prefix):]
                    break

            if not strategy or not category:
                continue

            if strategy != self.target_strategy:
                continue

            prompt_file = item_dir / f"04_enhanced_prompt_{self.enhancement_type}.txt"

            if prompt_file.exists():
                try:
                    with open(prompt_file, 'r', encoding='utf-8') as f:
                        prompt_text = f.read()

                    normalized_cat = self.normalize_name(category)
                    self.category_prompts[normalized_cat] = {
                        'template': prompt_text,
                        'category': category,
                        'strategy': strategy,
                        'enhancement_type': self.enhancement_type,
                        'file': str(prompt_file)
                    }

                    print(f"    ✓ Loaded: {category}")
                    loaded_count += 1

                except Exception as e:
                    print(f"    ⚠️  Error loading {prompt_file}: {e}")

        if loaded_count == 0:
            print(f"    ⚠️  No enhanced prompts found for {self.target_strategy}")
        else:
            print(f"  ✓ Total enhanced prompts loaded: {loaded_count}")

    def get_enhanced_prompt(self, category: str, answer_type: str = None) -> Optional[str]:
        normalized_cat = self.normalize_name(category)

        if normalized_cat in self.category_prompts:
            return self.category_prompts[normalized_cat]['template']

        for cat_key, prompt_data in self.category_prompts.items():
            if cat_key in normalized_cat or normalized_cat in cat_key:
                print(f"    ⚠️  Using partial match: {prompt_data['category']} for {category}")
                return prompt_data['template']

        return None

    def list_loaded_categories(self) -> List[str]:
        return [data['category'] for data in self.category_prompts.values()]


class BenchmarkTester:
    """Tests GPQA benchmark questions with enhanced prompts and O3 evaluation"""

    def __init__(self, api_key: str, data_base_dir: str = "./data",
                 enhancement_base_dir: str = None,
                 use_enhanced: bool = True, baseline_prompt_key: str = 'zero_shot_cot',
                 baseline_prompts_file: str = 'prompts.json',
                 force_self_refine: bool = False,
                 model: str = "gpt-4o",
                 checkpoint_dir: str = "./checkpoints",
                 checkpoint_interval: int = 10,
                 resume_from_checkpoint: str = None,
                 enhancement_type: str = 'specific',
                 target_strategy: str = 'zero_shot',
                 shuffle_choices: bool = True):
        """Initialize the tester"""
        self.client = OpenAI(api_key=api_key)
        self.use_enhanced = use_enhanced
        self.baseline_prompt_key = baseline_prompt_key
        self.force_self_refine = force_self_refine
        self.data_base_dir = Path(data_base_dir)
        self.model = model
        self.enhancement_type = enhancement_type
        self.target_strategy = target_strategy
        self.shuffle_choices = shuffle_choices

        # Checkpoint settings
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_interval = checkpoint_interval
        self.resume_from_checkpoint = resume_from_checkpoint
        self.current_checkpoint_file = None
        self._interrupted = False

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize prompt managers
        self.baseline_mgr = BaselinePromptsManager(baseline_prompts_file)
        self.prompt_mgr = BenchmarkPromptManager(
            enhancement_base_dir,
            use_enhanced,
            enhancement_type=enhancement_type,
            target_strategy=target_strategy
        )

        self.use_self_refine = force_self_refine or self.baseline_mgr.is_self_refine_prompt(baseline_prompt_key)
        self.use_self_consistency = self.baseline_mgr.is_self_consistency_prompt(baseline_prompt_key)

        self.vote_agent = MajorityVoteAgent(
            client=self.client,
            use_llm=True,
            model="gpt-4o"
        )

        mode = []
        if self.use_enhanced:
            mode.append(f"Enhanced Prompts ({enhancement_type})")
        if self.use_self_consistency:
            mode.append("Self-Consistency + MajorityVoteAgent")
        if self.use_self_refine:
            mode.append("Self-Refine")
        if not mode:
            mode.append(f"Baseline ({baseline_prompt_key})")

        print(f"\n  🎯 Mode: {' + '.join(mode)}")
        print(f"  🤖 Model: {self.model}")
        print(f"  💾 Checkpoint interval: every {checkpoint_interval} questions")
        print(f"  📁 Checkpoint directory: {self.checkpoint_dir}")
        print(f"  🔀 Shuffle choices: {shuffle_choices}")
        if self.use_enhanced:
            print(f"  📝 Enhancement type: {enhancement_type}")
            print(f"  🎯 Target strategy: {target_strategy}")

        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        def signal_handler(signum, frame):
            print(f"\n\n⚠️  Interrupt received! Saving checkpoint before exit...")
            self._interrupted = True

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def save_checkpoint(self, checkpoint_data: CheckpointData, is_final: bool = False):
        if is_final:
            filename = f"checkpoint_FINAL_{checkpoint_data.timestamp}.json"
        else:
            filename = f"checkpoint_{checkpoint_data.timestamp}.json"

        filepath = self.checkpoint_dir / filename
        self.current_checkpoint_file = filepath

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data.to_dict(), f, indent=2, ensure_ascii=False)

        print(f"  💾 Checkpoint saved: {filepath}")
        return filepath

    def load_checkpoint(self, checkpoint_path: str) -> CheckpointData:
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        checkpoint = CheckpointData.from_dict(data)
        print(f"  📂 Loaded checkpoint: {checkpoint_path}")
        print(f"     - Progress: {checkpoint.current_index}/{checkpoint.total_questions}")
        print(f"     - Completed: {len(checkpoint.completed_question_ids)} questions")
        print(f"     - Elapsed time: {checkpoint.elapsed_time:.1f}s")

        return checkpoint

    def load_gpqa_data(self, data_file: str) -> List[BenchmarkQuestion]:
        """Load GPQA data from CSV file"""
        questions = []

        data_path = Path(data_file)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")

        print(f"\n  Loading GPQA data from: {data_file}")

        df = pd.read_csv(data_file)
        print(f"  ✓ Loaded {len(df)} rows from CSV")

        for idx, row in df.iterrows():
            question = self._convert_gpqa_row_to_benchmark_question(row, idx)
            if question:
                questions.append(question)

        print(f"  ✓ Converted {len(questions)} questions")
        return questions

    def _convert_gpqa_row_to_benchmark_question(self, row: pd.Series, index: int) -> Optional[BenchmarkQuestion]:
        """Convert a GPQA CSV row to BenchmarkQuestion"""
        try:
            question_text = str(row.get('Question', '')).strip()
            correct_answer = str(row.get('Correct Answer', '')).strip()

            incorrect_answers = [
                str(row.get('Incorrect Answer 1', '')).strip(),
                str(row.get('Incorrect Answer 2', '')).strip(),
                str(row.get('Incorrect Answer 3', '')).strip()
            ]
            # Filter out empty answers
            incorrect_answers = [a for a in incorrect_answers if a and a != 'nan']

            if not question_text or not correct_answer:
                print(f"  ⚠️ Skipping row {index}: missing question or answer")
                return None

            # Get metadata
            category = str(row.get('High-level domain', 'Unknown')).strip()
            subdomain = str(row.get('Subdomain', '')).strip()
            difficulty = str(row.get("Writer's Difficulty Estimate", '')).strip()
            record_id = str(row.get('Record ID', f'gpqa_{index}')).strip()
            explanation = str(row.get('Explanation', '')).strip()

            # Create shuffled answer choices
            all_answers = [correct_answer] + incorrect_answers

            if self.shuffle_choices:
                random.shuffle(all_answers)

            # Map to letters
            letters = ['A', 'B', 'C', 'D']
            answer_choices = {}
            correct_letter = ''

            for i, answer in enumerate(all_answers[:4]):  # Max 4 choices
                letter = letters[i]
                answer_choices[letter] = answer
                if answer == correct_answer:
                    correct_letter = letter

            return BenchmarkQuestion(
                question_id=record_id,
                question=question_text,
                image='',
                answer=correct_answer,
                answer_type='multipleChoice',
                category=category,
                subject=subdomain,
                author='',
                rationale=explanation,
                difficulty=difficulty,
                incorrect_answers=incorrect_answers,
                answer_choices=answer_choices,
                correct_letter=correct_letter
            )

        except Exception as e:
            print(f"  ⚠️ Error converting row {index}: {e}")
            return None

    def format_choices(self, answer_choices: Dict[str, str]) -> str:
        """Format answer choices for display in prompt"""
        lines = []
        for letter in ['A', 'B', 'C', 'D']:
            if letter in answer_choices:
                lines.append(f"{letter}. {answer_choices[letter]}")
        return "\n".join(lines)

    def get_prompt_for_question(self, question: BenchmarkQuestion) -> str:
        """Get the appropriate prompt for a question"""
        if self.use_enhanced:
            enhanced_prompt = self.prompt_mgr.get_enhanced_prompt(
                question.category,
                question.answer_type
            )
            if enhanced_prompt:
                print(f"    → Using enhanced prompt ({self.enhancement_type}) for {question.category}")
                return enhanced_prompt

        baseline_prompt = self.baseline_mgr.get_prompt_template(self.baseline_prompt_key)
        if baseline_prompt:
            print(f"    → Using baseline prompt: {self.baseline_prompt_key}")
            return baseline_prompt

        # Ultimate fallback
        return """Answer the following multiple choice question. Think step by step.

Question: {question}

{choices}

Provide your final answer as a single letter (A, B, C, or D).

Final Answer:"""

    def call_model(self, messages: List[Dict], temperature: float = 0,
                   max_tokens: int = 3000) -> str:
        """Call OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  ⚠️  API Error: {e}")
            return ""

    def test_question_with_self_consistency(self, question: BenchmarkQuestion,
                                            num_samples: int = 5) -> TestResult:
        """Test question using self-consistency with MajorityVoteAgent"""
        self.vote_agent.reset()
        self.vote_agent.set_question(question.question, question.answer_type)

        prompt_template = self.get_prompt_for_question(question)
        choices_text = self.format_choices(question.answer_choices)
        prompt = prompt_template.replace("{question}", question.question).replace("{choices}", choices_text)

        predictions = []

        print(f"  🔄 Sampling {num_samples} predictions...")
        for i in range(num_samples):
            messages = [{"role": "user", "content": prompt}]
            response = self.call_model(messages, temperature=0.7)
            answer = self.extract_answer(response, question.answer_type)
            predictions.append(answer)
            self.vote_agent.add_prediction(answer, response[:500])
            print(f"    Sample {i + 1}: {answer}")

        print(f"  🤖 Agent analyzing predictions...")
        final_answer, counts, confidence, agent_analysis = self.vote_agent.get_majority_answer()

        print(f"  🗳️  Voting results: {counts}")
        print(f"  ✓ Winner: {final_answer} (confidence: {confidence:.2%})")

        if confidence < 0.6:
            print(f"  ⚠️  Low confidence, analyzing disagreement...")
            disagreement = self.vote_agent.analyze_disagreement()
            if 'analysis' in disagreement:
                agent_analysis += f"\n\nDisagreement Analysis:\n{disagreement['analysis']}"

        # Evaluate
        is_correct = final_answer.upper() == question.correct_letter.upper()

        return TestResult(
            question_id=question.question_id,
            question=question.question,
            correct_answer=question.answer,
            predicted_answer=question.answer_choices.get(final_answer.upper(), final_answer),
            is_correct=is_correct,
            reasoning=f"Self-consistency with {num_samples} samples + MajorityVoteAgent",
            answer_type=question.answer_type,
            category=question.category,
            subject=question.subject,
            has_image=False,
            evaluation_reasoning=f"Predicted: {final_answer}, Correct: {question.correct_letter}",
            all_predictions=predictions,
            prediction_counts=counts,
            confidence=confidence,
            agent_analysis=agent_analysis,
            difficulty=question.difficulty,
            correct_letter=question.correct_letter,
            predicted_letter=final_answer,
            answer_choices=question.answer_choices
        )

    def test_question_with_self_refine(self, question: BenchmarkQuestion) -> TestResult:
        """Test question using self-refine"""
        prompt_template = self.get_prompt_for_question(question)
        choices_text = self.format_choices(question.answer_choices)
        initial_prompt = prompt_template.replace("{question}", question.question).replace("{choices}", choices_text)

        print(f"  📝 Getting initial answer...")
        messages = [{"role": "user", "content": initial_prompt}]
        initial_response = self.call_model(messages, temperature=0)
        initial_answer = self.extract_answer(initial_response, question.answer_type)
        print(f"    Initial: {initial_answer}")

        print(f"  🔧 Refining answer...")
        refine_prompt = f"""You previously answered this question with: {initial_answer}

Please review your answer and reasoning. Consider:
1. Did you interpret the question correctly?
2. Did you consider all the answer choices carefully?
3. Is there any scientific principle you might have overlooked?
4. Are you confident in your answer?

Question: {question.question}

{choices_text}

After reflection, provide your final answer as a single letter (A, B, C, or D).

Final Answer:"""

        messages.append({"role": "assistant", "content": initial_response})
        messages.append({"role": "user", "content": refine_prompt})

        refined_response = self.call_model(messages, temperature=0)
        refined_answer = self.extract_answer(refined_response, question.answer_type)
        print(f"    Refined: {refined_answer}")

        was_refined = initial_answer.upper() != refined_answer.upper()

        is_correct = refined_answer.upper() == question.correct_letter.upper()

        return TestResult(
            question_id=question.question_id,
            question=question.question,
            correct_answer=question.answer,
            predicted_answer=question.answer_choices.get(refined_answer.upper(), refined_answer),
            is_correct=is_correct,
            reasoning=refined_response,
            answer_type=question.answer_type,
            category=question.category,
            subject=question.subject,
            has_image=False,
            evaluation_reasoning=f"Predicted: {refined_answer}, Correct: {question.correct_letter}",
            initial_answer=initial_answer,
            was_refined=was_refined,
            difficulty=question.difficulty,
            correct_letter=question.correct_letter,
            predicted_letter=refined_answer,
            answer_choices=question.answer_choices
        )

    def test_question(self, question: BenchmarkQuestion) -> TestResult:
        """Test a single question"""
        if self.use_self_consistency:
            sampling_params = self.baseline_mgr.get_sampling_params(self.baseline_prompt_key)
            num_samples = sampling_params.get('num_samples', 5)
            return self.test_question_with_self_consistency(question, num_samples)

        if self.use_self_refine:
            return self.test_question_with_self_refine(question)

        # Standard single-pass testing
        prompt_template = self.get_prompt_for_question(question)
        choices_text = self.format_choices(question.answer_choices)
        prompt = prompt_template.replace("{question}", question.question).replace("{choices}", choices_text)

        messages = [{"role": "user", "content": prompt}]
        response = self.call_model(messages)
        predicted_answer = self.extract_answer(response, question.answer_type)

        is_correct = predicted_answer.upper() == question.correct_letter.upper()

        return TestResult(
            question_id=question.question_id,
            question=question.question,
            correct_answer=question.answer,
            predicted_answer=question.answer_choices.get(predicted_answer.upper(), predicted_answer),
            is_correct=is_correct,
            reasoning=response,
            answer_type=question.answer_type,
            category=question.category,
            subject=question.subject,
            has_image=False,
            evaluation_reasoning=f"Predicted: {predicted_answer}, Correct: {question.correct_letter}",
            difficulty=question.difficulty,
            correct_letter=question.correct_letter,
            predicted_letter=predicted_answer,
            answer_choices=question.answer_choices
        )

    def extract_answer(self, response: str, answer_type: str) -> str:
        """Extract answer letter from model response"""
        if not response:
            return ""

        # For multiple choice, extract the letter
        patterns = [
            r'(?:final answer|answer)(?:\s*is)?(?:\s*:)?\s*\(?([A-D])\)?',
            r'\b([A-D])\s*(?:is|would be)(?:\s+the)?\s+(?:correct|answer)',
            r'(?:^|\n)\s*\(?([A-D])\)?[\s.:)]',
            r'\*\*([A-D])\*\*',
            r'(?:select|choose|pick)\s+(?:option\s+)?([A-D])',
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).upper()

        # Look for standalone letter at end
        lines = response.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            if len(line) <= 3:
                letter_match = re.search(r'([A-D])', line, re.IGNORECASE)
                if letter_match:
                    return letter_match.group(1).upper()

        # Last resort: find any A-D
        letters = re.findall(r'\b([A-D])\b', response)
        if letters:
            return letters[-1].upper()

        return ""

    def save_failed_questions(self, results: List[TestResult], output_dir: str):
        """Save failed questions organized by category and difficulty"""
        os.makedirs(output_dir, exist_ok=True)

        failures_by_category = defaultdict(lambda: defaultdict(list))

        for result in results:
            if not result.is_correct:
                failure_data = {
                    'question_id': result.question_id,
                    'question': result.question[:500],
                    'correct_answer': result.correct_answer,
                    'correct_letter': result.correct_letter,
                    'predicted_answer': result.predicted_answer,
                    'predicted_letter': result.predicted_letter,
                    'reasoning': result.reasoning[:500] if result.reasoning else '',
                    'evaluation': result.evaluation_reasoning,
                    'difficulty': result.difficulty,
                    'subject': result.subject,
                    'answer_choices': result.answer_choices
                }
                if result.agent_analysis:
                    failure_data['agent_analysis'] = result.agent_analysis
                if result.all_predictions:
                    failure_data['all_predictions'] = result.all_predictions
                if result.confidence is not None:
                    failure_data['confidence'] = result.confidence

                failures_by_category[result.category][result.difficulty or 'unknown'].append(failure_data)

        for category, difficulties in failures_by_category.items():
            for difficulty, failures in difficulties.items():
                safe_category = category.replace(' ', '_').replace('/', '_').replace(',', '_')
                safe_difficulty = str(difficulty).replace(' ', '_')
                filename = f"{safe_category}_{safe_difficulty}_failures.json"
                filepath = os.path.join(output_dir, filename)

                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump({
                        'category': category,
                        'difficulty': difficulty,
                        'total_failures': len(failures),
                        'failures': failures
                    }, f, indent=2, ensure_ascii=False)

                print(f"  💾 Saved {len(failures)} {category} (difficulty {difficulty}) failures to {filename}")

        # Save all results summary
        all_results_data = []
        for r in results:
            all_results_data.append({
                'question_id': r.question_id,
                'is_correct': r.is_correct,
                'predicted_letter': r.predicted_letter,
                'correct_letter': r.correct_letter,
                'category': r.category,
                'subject': r.subject,
                'difficulty': r.difficulty,
                'confidence': r.confidence,
                'evaluation': r.evaluation_reasoning
            })

        with open(os.path.join(output_dir, 'all_results.json'), 'w') as f:
            json.dump(all_results_data, f, indent=2)

        print(f"\n  💾 Results saved to {output_dir}/")

    def print_category_accuracy_summary(self, results: List[TestResult]):
        """Print detailed category accuracy summary"""
        print(f"\n{'=' * 80}")
        print("CATEGORY ACCURACY SUMMARY (High-level Domain)")
        print(f"{'=' * 80}")

        category_results = defaultdict(list)
        for result in results:
            category_results[result.category].append(result)

        category_stats = []
        for category, cat_results in category_results.items():
            total = len(cat_results)
            correct = sum(1 for r in cat_results if r.is_correct)
            accuracy = correct / total if total > 0 else 0
            category_stats.append({
                'category': category,
                'total': total,
                'correct': correct,
                'incorrect': total - correct,
                'accuracy': accuracy
            })

        category_stats.sort(key=lambda x: x['total'], reverse=True)

        print(f"\n{'Category':<30} {'Total':>8} {'Correct':>8} {'Wrong':>8} {'Accuracy':>10}")
        print(f"{'-' * 70}")

        for stats in category_stats:
            print(f"{stats['category'][:30]:<30} {stats['total']:>8} {stats['correct']:>8} {stats['incorrect']:>8} {stats['accuracy']:>9.1%}")

        total_all = sum(s['total'] for s in category_stats)
        correct_all = sum(s['correct'] for s in category_stats)
        overall_accuracy = correct_all / total_all if total_all > 0 else 0

        print(f"{'-' * 70}")
        print(f"{'TOTAL':<30} {total_all:>8} {correct_all:>8} {total_all - correct_all:>8} {overall_accuracy:>9.1%}")
        print(f"{'=' * 70}")

        # Subdomain breakdown
        subdomain_results = defaultdict(list)
        for result in results:
            if result.subject:
                subdomain_results[result.subject].append(result)

        if subdomain_results:
            print(f"\n{'=' * 80}")
            print("SUBDOMAIN ACCURACY SUMMARY")
            print(f"{'=' * 80}")

            subdomain_stats = []
            for subdomain, sub_results in subdomain_results.items():
                total = len(sub_results)
                correct = sum(1 for r in sub_results if r.is_correct)
                accuracy = correct / total if total > 0 else 0
                subdomain_stats.append({
                    'subdomain': subdomain,
                    'total': total,
                    'correct': correct,
                    'accuracy': accuracy
                })

            subdomain_stats.sort(key=lambda x: x['total'], reverse=True)

            print(f"\n{'Subdomain':<40} {'Total':>8} {'Correct':>8} {'Accuracy':>10}")
            print(f"{'-' * 70}")

            for stats in subdomain_stats[:20]:  # Top 20
                print(f"{stats['subdomain'][:40]:<40} {stats['total']:>8} {stats['correct']:>8} {stats['accuracy']:>9.1%}")

            if len(subdomain_stats) > 20:
                print(f"  ... and {len(subdomain_stats) - 20} more subdomains")

            print(f"{'=' * 70}")

        # Difficulty breakdown
        difficulty_results = defaultdict(list)
        for result in results:
            if result.difficulty:
                difficulty_results[result.difficulty].append(result)

        if difficulty_results:
            print(f"\n{'=' * 80}")
            print("DIFFICULTY ACCURACY SUMMARY")
            print(f"{'=' * 80}")

            print(f"\n{'Difficulty':<50} {'Total':>8} {'Correct':>8} {'Accuracy':>10}")
            print(f"{'-' * 80}")

            for difficulty in sorted(difficulty_results.keys()):
                diff_results = difficulty_results[difficulty]
                total = len(diff_results)
                correct = sum(1 for r in diff_results if r.is_correct)
                accuracy = correct / total if total > 0 else 0
                print(f"{str(difficulty)[:50]:<50} {total:>8} {correct:>8} {accuracy:>9.1%}")

            print(f"{'=' * 80}")

    def run_test(self, data_file: str = None, start_index: int = 0,
                 end_index: int = None, categories: List[str] = None) -> Dict:
        """Run test on GPQA data with checkpointing support"""
        print(f"\n{'=' * 80}")
        print(f"LOADING GPQA DATA")
        print(f"{'=' * 80}")

        all_results = []
        completed_ids = set()
        elapsed_time = 0.0
        resume_index = 0

        if self.resume_from_checkpoint:
            try:
                checkpoint = self.load_checkpoint(self.resume_from_checkpoint)
                all_results = [TestResult.from_dict(r) for r in checkpoint.results]
                completed_ids = set(checkpoint.completed_question_ids)
                elapsed_time = checkpoint.elapsed_time
                resume_index = checkpoint.current_index
                data_file = checkpoint.data_file
                start_index = checkpoint.start_index
                end_index = checkpoint.end_index
                print(f"  ✓ Resuming from question {resume_index}")
            except Exception as e:
                print(f"  ⚠️ Failed to load checkpoint: {e}")
                print(f"  Starting fresh...")

        if data_file is None:
            data_file = str(self.data_base_dir / "gpqa_diamond_train.csv")

        questions = self.load_gpqa_data(data_file)

        if categories:
            questions = [q for q in questions if q.category in categories]
            print(f"  ✓ Filtered to categories {categories}: {len(questions)} questions")

        total_before_slice = len(questions)
        if end_index is not None:
            questions = questions[start_index:end_index]
        else:
            questions = questions[start_index:]

        if start_index > 0 or end_index is not None:
            print(f"  ✓ Index range [{start_index}:{end_index or total_before_slice}]: {len(questions)} questions")

        if not questions:
            print(f"  ✗ No questions to test!")
            return {}

        if completed_ids:
            questions_to_run = [(i, q) for i, q in enumerate(questions) if q.question_id not in completed_ids]
            print(f"  ✓ Skipping {len(completed_ids)} already completed questions")
        else:
            questions_to_run = list(enumerate(questions))

        print(f"\n{'=' * 80}")
        print(f"TESTING {len(questions_to_run)} QUESTIONS (of {len(questions)} total)")
        print(f"{'=' * 80}")

        start_time = time.time()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        config = {
            'model': self.model,
            'use_enhanced': self.use_enhanced,
            'baseline_prompt_key': self.baseline_prompt_key,
            'use_self_consistency': self.use_self_consistency,
            'use_self_refine': self.use_self_refine,
            'categories': categories,
            'enhancement_type': self.enhancement_type,
            'target_strategy': self.target_strategy,
            'shuffle_choices': self.shuffle_choices
        }

        try:
            for idx, (original_idx, question) in enumerate(questions_to_run):
                if self._interrupted:
                    print(f"\n⚠️  Saving checkpoint due to interruption...")
                    break

                print(f"\n[{len(all_results) + 1}/{len(questions)}] Question {question.question_id}")
                print(f"  Category: {question.category}")
                print(f"  Subdomain: {question.subject}")
                print(f"  Difficulty: {question.difficulty}")
                print(f"  Question: {question.question[:80]}...")

                try:
                    result = self.test_question(question)
                    all_results.append(result)
                    completed_ids.add(question.question_id)

                    status = '✓ CORRECT' if result.is_correct else '✗ INCORRECT'
                    print(f"  {status}")
                    print(f"  Predicted: {result.predicted_letter} - {result.predicted_answer[:50]}...")
                    print(f"  Correct: {result.correct_letter} - {result.correct_answer[:50]}...")

                except Exception as e:
                    print(f"  ✗ Error: {e}")
                    import traceback
                    traceback.print_exc()

                current_elapsed = elapsed_time + (time.time() - start_time)
                if (len(all_results) % self.checkpoint_interval == 0) or self._interrupted:
                    checkpoint_data = CheckpointData(
                        timestamp=timestamp,
                        data_file=data_file,
                        start_index=start_index,
                        end_index=end_index,
                        current_index=original_idx + 1,
                        total_questions=len(questions),
                        completed_question_ids=list(completed_ids),
                        results=[r.to_dict() for r in all_results],
                        config=config,
                        elapsed_time=current_elapsed
                    )
                    self.save_checkpoint(checkpoint_data)

        except KeyboardInterrupt:
            print(f"\n\n⚠️  KeyboardInterrupt! Saving final checkpoint...")
            self._interrupted = True

        total_time = elapsed_time + (time.time() - start_time)

        checkpoint_data = CheckpointData(
            timestamp=timestamp,
            data_file=data_file,
            start_index=start_index,
            end_index=end_index,
            current_index=len(questions),
            total_questions=len(questions),
            completed_question_ids=list(completed_ids),
            results=[r.to_dict() for r in all_results],
            config=config,
            elapsed_time=total_time
        )
        self.save_checkpoint(checkpoint_data, is_final=not self._interrupted)

        total = len(all_results)
        correct = sum(1 for r in all_results if r.is_correct)
        accuracy = correct / total if total > 0 else 0

        output_dir = f"gpqa_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.save_failed_questions(all_results, output_dir)

        print(f"\n{'=' * 80}")
        print(f"TEST SUMMARY")
        print(f"{'=' * 80}")
        if self._interrupted:
            print(f"⚠️  RUN INTERRUPTED - Partial results")
        print(f"Total Questions: {total}")
        print(f"Correct: {correct}")
        print(f"Incorrect: {total - correct}")
        print(f"Overall Accuracy: {accuracy:.2%}")
        print(f"Total Time: {total_time:.1f}s ({total_time / total:.1f}s per question)" if total > 0 else "")
        print(f"Checkpoint saved: {self.current_checkpoint_file}")

        if self.use_self_consistency:
            confidences = [r.confidence for r in all_results if r.confidence is not None]
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                print(f"Average Confidence: {avg_confidence:.2%}")

                high_conf = [r for r in all_results if r.confidence and r.confidence >= 0.6]
                low_conf = [r for r in all_results if r.confidence and r.confidence < 0.6]

                if high_conf:
                    high_acc = sum(1 for r in high_conf if r.is_correct) / len(high_conf)
                    print(f"High Confidence (>=60%) Accuracy: {high_acc:.2%} ({len(high_conf)} questions)")
                if low_conf:
                    low_acc = sum(1 for r in low_conf if r.is_correct) / len(low_conf)
                    print(f"Low Confidence (<60%) Accuracy: {low_acc:.2%} ({len(low_conf)} questions)")

        self.print_category_accuracy_summary(all_results)

        if self._interrupted:
            print(f"\n💡 To resume this run, use:")
            print(f"   python benchmark_gpqa.py --resume-from {self.current_checkpoint_file}")

        return {
            'overall_accuracy': accuracy,
            'total_questions': total,
            'correct': correct,
            'incorrect': total - correct,
            'time_seconds': total_time,
            'results_dir': output_dir,
            'checkpoint_file': str(self.current_checkpoint_file),
            'interrupted': self._interrupted
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='GPQA Benchmark Testing with Enhanced Prompts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Test on train set (baseline only)
  python benchmark_gpqa.py --data-file ./data/gpqa_diamond_train.csv --no-enhance

  # Test on test set with index range
  python benchmark_gpqa.py --data-file ./data/gpqa_diamond_test.csv --start-index 0 --end-index 10

  # Test with self-consistency
  python benchmark_gpqa.py --data-file ./data/gpqa_diamond_train.csv --prompt-type zero_shot_self_consistency

  # Run with enhanced prompts (specific type)
  python benchmark_gpqa.py --data-file ./data/gpqa_diamond_train.csv \\
    --enhanced-prompt-dir ./gpqa_enhanced_prompts/enhanced_20251224_120000 \\
    --enhancement-type specific

  # Run ALL: baseline + all three enhancement types for comparison
  python benchmark_gpqa.py --data-file ./data/gpqa_diamond_train.csv \\
    --enhanced-prompt-dir ./gpqa_enhanced_prompts/enhanced_20251224_120000 \\
    --enhancement-type all

  # Run ALL enhancements only (no baseline)
  python benchmark_gpqa.py --data-file ./data/gpqa_diamond_train.csv \\
    --enhanced-prompt-dir ./gpqa_enhanced_prompts/enhanced_20251224_120000 \\
    --enhancement-type all_no_baseline

  # Resume from checkpoint
  python benchmark_gpqa.py --resume-from ./checkpoints/checkpoint_20251219_120000.json

  # Test specific categories
  python benchmark_gpqa.py --data-file ./data/gpqa_diamond_train.csv --categories Physics Chemistry
        '''
    )

    parser.add_argument('--data-dir', default='./data',
                        help='Base directory for data files')
    parser.add_argument('--data-file', type=str, default='gpqa_extended_test.csv',
                        help='Path to GPQA data file (CSV)')
    parser.add_argument('--start-index', type=int, default=0,
                        help='Starting index for questions (0-based)')
    parser.add_argument('--end-index', type=int, default=None,
                        help='Ending index for questions (exclusive)')

    parser.add_argument('--enhanced-prompt-dir', type=str, default='gpqa_enhanced_prompts/enhanced_20260105_014521',
                        help='Directory containing enhanced prompts')
    parser.add_argument('--enhancement-type', type=str, default='all',
                        choices=['concise', 'specific', 'reasoning', 'all', 'all_no_baseline'],
                        help='Type of enhancement to use ("all" = baseline + all three, "all_no_baseline" = only all three enhancements)')
    parser.add_argument('--target-strategy', type=str, default='zero_shot_cot',
                        choices=['zero_shot', 'zero_shot_cot', 'self_refine'],
                        help='Which strategy\'s enhanced prompts to load')
    parser.add_argument('--no-enhance', action='store_true', default=False,
                        help='Disable enhanced prompts, use only baseline')

    parser.add_argument('--prompt-type', type=str, default='zero_shot_self_consistency',
                        help='Baseline prompt type')
    parser.add_argument('--use-self-refine', action='store_true', default=False,
                        help='Use self-refine')
    parser.add_argument('--baseline-prompts', type=str, default='prompts.json',
                        help='Path to baseline prompts JSON file')
    parser.add_argument('--model', type=str, default='gpt-4o',
                        help='Model to use for generation')
    parser.add_argument('--categories', nargs='+',
                        help='Filter by specific high-level domains (Physics, Chemistry, Biology)')
    parser.add_argument('--api-key', type=str,
                        default=os.getenv('OPENAI_API_KEY'),
                        help='OpenAI API key')

    parser.add_argument('--no-shuffle', action='store_true', default=False,
                        help='Do not shuffle answer choices')

    parser.add_argument('--checkpoint-dir', type=str, default='./gpqa_checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--checkpoint-interval', type=int, default=10,
                        help='Save checkpoint every N questions')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Path to checkpoint file to resume from')

    args = parser.parse_args()

    api_key = args.api_key
    if not api_key:
        print("✗ Error: OpenAI API key not found!")
        print("  Set OPENAI_API_KEY environment variable or use --api-key argument")
        return

    # Default data file
    if args.data_file is None:
        args.data_file = os.path.join(args.data_dir, 'gpqa_diamond_train.csv')

    if not args.resume_from and not os.path.exists(args.data_file):
        print(f"\n✗ Error: Data file '{args.data_file}' not found!")
        return

    use_enhanced = not args.no_enhance and args.enhanced_prompt_dir is not None

    # Check enhancement directory if using enhanced prompts
    if use_enhanced and args.enhanced_prompt_dir:
        from pathlib import Path
        enhancement_path = Path(args.enhanced_prompt_dir)
        if not enhancement_path.exists():
            print(f"\n  Warning: Enhancement directory '{args.enhanced_prompt_dir}' not found!")
            print(f"  Using baseline prompts only.")
            use_enhanced = False

    print("\n" + "=" * 80)
    print("GPQA BENCHMARK TESTING")
    print("=" * 80)
    if args.resume_from:
        print(f"Resuming from: {args.resume_from}")
    else:
        print(f"Data file: {args.data_file}")
    print(f"Data directory: {args.data_dir}")
    print(f"Enhanced prompts: {'Enabled' if use_enhanced else 'Disabled'}")
    if use_enhanced:
        print(f"  - Directory: {args.enhanced_prompt_dir}")
        if args.enhancement_type == 'all':
            type_desc = " (will run baseline + all three)"
        elif args.enhancement_type == 'all_no_baseline':
            type_desc = " (will run all three, no baseline)"
        else:
            type_desc = ""
        print(f"  - Type: {args.enhancement_type}" + type_desc)
        print(f"  - Strategy: {args.target_strategy}")
    print(f"Baseline prompt: {args.prompt_type}")
    print(f"Model: {args.model}")
    print(f"Index range: [{args.start_index}:{args.end_index or 'end'}]")
    print(f"Shuffle choices: {not args.no_shuffle}")
    print(f"Checkpoint interval: {args.checkpoint_interval}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    if args.categories:
        print(f"Categories: {args.categories}")
    print("=" * 80)

    # Check if running all enhancement types
    if args.enhancement_type in ['all', 'all_no_baseline'] and use_enhanced:
        include_baseline = args.enhancement_type == 'all'

        if include_baseline:
            print("\n🔄 Running BASELINE + ALL THREE enhancement types for comparison...")
        else:
            print("\n🔄 Running ALL THREE enhancement types (no baseline)...")

        all_enhancement_results = {}

        # Run baseline only if 'all' (not 'all_no_baseline')
        if include_baseline:
            print(f"\n{'=' * 80}")
            print(f"RUNNING BASELINE (NO ENHANCEMENT)")
            print(f"{'=' * 80}")

            try:
                tester = BenchmarkTester(
                    api_key=api_key,
                    data_base_dir=args.data_dir,
                    enhancement_base_dir=None,
                    use_enhanced=False,
                    baseline_prompt_key=args.prompt_type,
                    baseline_prompts_file=args.baseline_prompts,
                    force_self_refine=args.use_self_refine,
                    model=args.model,
                    checkpoint_dir=args.checkpoint_dir,
                    checkpoint_interval=args.checkpoint_interval,
                    resume_from_checkpoint=None,
                    enhancement_type='none',
                    target_strategy=args.target_strategy,
                    shuffle_choices=not args.no_shuffle
                )

                results = tester.run_test(
                    data_file=args.data_file,
                    start_index=args.start_index,
                    end_index=args.end_index,
                    categories=args.categories
                )

                all_enhancement_results['baseline'] = results

            except Exception as e:
                print(f"\n✗ Error with baseline: {e}")
                all_enhancement_results['baseline'] = {'error': str(e)}

        # Run each enhancement type
        for enh_type in ['concise', 'specific', 'reasoning']:
            print(f"\n{'=' * 80}")
            print(f"RUNNING WITH ENHANCEMENT TYPE: {enh_type.upper()}")
            print(f"{'=' * 80}")

            try:
                tester = BenchmarkTester(
                    api_key=api_key,
                    data_base_dir=args.data_dir,
                    enhancement_base_dir=args.enhanced_prompt_dir,
                    use_enhanced=True,
                    baseline_prompt_key=args.prompt_type,
                    baseline_prompts_file=args.baseline_prompts,
                    force_self_refine=args.use_self_refine,
                    model=args.model,
                    checkpoint_dir=args.checkpoint_dir,
                    checkpoint_interval=args.checkpoint_interval,
                    resume_from_checkpoint=None,  # Don't resume for comparison runs
                    enhancement_type=enh_type,
                    target_strategy=args.target_strategy,
                    shuffle_choices=not args.no_shuffle
                )

                results = tester.run_test(
                    data_file=args.data_file,
                    start_index=args.start_index,
                    end_index=args.end_index,
                    categories=args.categories
                )

                all_enhancement_results[enh_type] = results

            except Exception as e:
                print(f"\n✗ Error with {enh_type}: {e}")
                import traceback
                traceback.print_exc()
                all_enhancement_results[enh_type] = {'error': str(e)}

        # Print comparison summary
        print(f"\n{'=' * 80}")
        print("ENHANCEMENT TYPE COMPARISON SUMMARY")
        print(f"{'=' * 80}")

        # Get baseline accuracy for comparison (if available)
        baseline_acc = all_enhancement_results.get('baseline', {}).get('overall_accuracy', None)

        # Determine which types to show
        if include_baseline:
            types_to_show = ['baseline', 'concise', 'specific', 'reasoning']
        else:
            types_to_show = ['concise', 'specific', 'reasoning']

        if include_baseline:
            print(f"\n{'Type':<15} {'Accuracy':>12} {'Correct':>10} {'Total':>10} {'vs Baseline':>15}")
        else:
            print(f"\n{'Type':<15} {'Accuracy':>12} {'Correct':>10} {'Total':>10}")
        print(f"{'-' * 65}")

        for enh_type in types_to_show:
            results = all_enhancement_results.get(enh_type, {})
            if 'error' in results:
                if include_baseline:
                    print(f"{enh_type:<15} {'ERROR':>12} {'-':>10} {'-':>10} {'-':>15}")
                else:
                    print(f"{enh_type:<15} {'ERROR':>12} {'-':>10} {'-':>10}")
            else:
                acc = results.get('overall_accuracy', 0)
                correct = results.get('correct', 0)
                total = results.get('total_questions', 0)

                if include_baseline:
                    if enh_type == 'baseline':
                        diff_str = "-"
                    else:
                        diff = acc - baseline_acc if baseline_acc else 0
                        diff_str = f"{diff:+.2%}"
                    print(f"{enh_type:<15} {acc:>11.2%} {correct:>10} {total:>10} {diff_str:>15}")
                else:
                    print(f"{enh_type:<15} {acc:>11.2%} {correct:>10} {total:>10}")

        print(f"{'=' * 65}")

        # Find best enhancement type (excluding baseline)
        enhancement_results = {k: v for k, v in all_enhancement_results.items()
                              if k != 'baseline' and 'error' not in v}

        if enhancement_results:
            best_type = max(
                [(k, v.get('overall_accuracy', 0)) for k, v in enhancement_results.items()],
                key=lambda x: x[1]
            )

            print(f"\n🏆 Best Enhancement: {best_type[0].upper()} ({best_type[1]:.2%})")

            if include_baseline and baseline_acc is not None:
                improvement = best_type[1] - baseline_acc
                print(f"📈 Improvement over baseline: {improvement:+.2%}")

        print(f"{'=' * 80}")

    else:
        # Single enhancement type run
        try:
            tester = BenchmarkTester(
                api_key=api_key,
                data_base_dir=args.data_dir,
                enhancement_base_dir=args.enhanced_prompt_dir if use_enhanced else None,
                use_enhanced=use_enhanced,
                baseline_prompt_key=args.prompt_type,
                baseline_prompts_file=args.baseline_prompts,
                force_self_refine=args.use_self_refine,
                model=args.model,
                checkpoint_dir=args.checkpoint_dir,
                checkpoint_interval=args.checkpoint_interval,
                resume_from_checkpoint=args.resume_from,
                enhancement_type=args.enhancement_type,
                target_strategy=args.target_strategy,
                shuffle_choices=not args.no_shuffle
            )

            results = tester.run_test(
                data_file=args.data_file if not args.resume_from else None,
                start_index=args.start_index,
                end_index=args.end_index,
                categories=args.categories
            )

            if results:
                print(f"\n✓ Test complete!")
                print(f"Overall Accuracy: {results['overall_accuracy']:.2%}")
                print(f"Results saved to: {results['results_dir']}")
                print(f"Checkpoint saved to: {results['checkpoint_file']}")

        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()