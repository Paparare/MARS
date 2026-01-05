"""
Unified Three-Agent Benchmark Enhancement System - GPQA Version
================================================================

Modified to work with failed questions from benchmark_gpqa.py output.
Keeps original output format.

Modified for Together AI API compatibility.

Input structure expected (from benchmark_gpqa.py):
    gpqa_results_YYYYMMDD_HHMMSS/
    ├── all_results.json
    ├── Physics_*_failures.json
    ├── Chemistry_*_failures.json
    └── Biology_*_failures.json

Enhanced workflow:
1. Individual Failure Analysis: Analyzes exact cause of error for each question
2. Type & Topic Extraction: Identifies question type and topics
3. Pattern Analysis: Groups failures by type and topic, analyzes patterns
4. Three Targeted Enhancements: Creates THREE enhancement types simultaneously:
   - CONCISE: Quick warnings and key points
   - SPECIFIC: Detailed guidance with verification steps
   - REASONING: Minimal hints for self-discovery
"""

import json
import os
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from together import Together
from pathlib import Path
from datetime import datetime
import time
import re
import glob


def create_client(api_key: Optional[str] = None) -> Together:
    """Create Together AI client"""
    if api_key:
        return Together(api_key=api_key)
    # Auth defaults to os.environ.get("TOGETHER_API_KEY")
    return Together()


def call_llm(client: Together, model: str, messages: List[Dict], temperature: float = 0.3,
             max_tokens: int = 800, stream: bool = False) -> str:
    """
    Call Together AI LLM API with support for both streaming and non-streaming responses.
    """
    if stream:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )

        full_content = ""

        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if hasattr(delta, 'content') and delta.content:
                full_content += delta.content

        return full_content
    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()


class PromptManager:
    """Manages loading and accessing prompts from JSON configuration"""

    def __init__(self, prompts_file: str = "prompts.json"):
        """Initialize the prompt manager with prompts from JSON file"""
        self.prompts = {}
        self.model_settings = {}
        self.metadata = {}
        self.prompts_file = prompts_file
        self.load_prompts(prompts_file)

    def load_prompts(self, prompts_file: str):
        """Load prompts from JSON file"""
        if not os.path.exists(prompts_file):
            print(f"  ⚠️ {prompts_file} not found. Creating default prompts file...")
            self._create_default_prompts(prompts_file)

        with open(prompts_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.prompts = data.get('prompts', {})
        self.model_settings = data.get('model_settings', {})
        self.metadata = data.get('metadata', {})

        print(f"  ✓ Loaded {len(self.prompts)} prompts from {prompts_file}")

    def _create_default_prompts(self, prompts_file: str):
        """Create default prompts.json for GPQA if it doesn't exist"""
        default_prompts = {
            "prompts": {
                "zero_shot": {
                    "name": "Zero-shot Direct Answer",
                    "description": "Zero-shot prompt for multiple choice",
                    "template": "Answer the following multiple choice question.\n\nQuestion: {question}\n\n{choices}\n\nAnswer: ",
                    "max_tokens": 100,
                    "variables": ["question", "choices"]
                },
                "zero_shot_cot": {
                    "name": "Zero-shot Chain-of-Thought",
                    "description": "Zero-shot prompt with step-by-step reasoning",
                    "template": "Answer the following multiple choice question. Think step by step.\n\nQuestion: {question}\n\n{choices}\n\nLet's approach this systematically:\nAnswer: ",
                    "max_tokens": 3000,
                    "variables": ["question", "choices"]
                },
                "self_refine": {
                    "name": "Self-Refine",
                    "description": "Self-refinement prompt for multiple choice",
                    "template": "Answer the following multiple choice question. After answering, verify your reasoning.\n\nQuestion: {question}\n\n{choices}\n\nAnswer: ",
                    "max_tokens": 3000,
                    "variables": ["question", "choices"]
                }
            },
            "model_settings": {
                "model": "moonshotai/Kimi-K2-Thinking",
                "temperature": 0.7,
                "default_max_tokens": 3000
            },
            "metadata": {
                "version": "1.0",
                "description": "Prompts for GPQA Enhancement",
                "created": datetime.now().strftime("%Y-%m-%d")
            }
        }

        with open(prompts_file, 'w', encoding='utf-8') as f:
            json.dump(default_prompts, f, indent=2)

    def get_prompt_template(self, prompt_key: str) -> str:
        """Get the prompt template for a given key"""
        if prompt_key not in self.prompts:
            # Return default template for multiple choice
            return "Answer the following multiple choice question.\n\nQuestion: {question}\n\n{choices}\n\nAnswer: "
        return self.prompts[prompt_key].get('template', '')


# Data structures
@dataclass
class IndividualFailureAnalysis:
    """Analysis of a single question's failure"""
    question_id: str
    question_text: str
    correct_answer: str
    model_answer: str
    correct_letter: str
    predicted_letter: str
    question_type: str  # e.g., "factual", "conceptual", "calculation", "application"
    topics: List[str]  # e.g., ["quantum mechanics", "energy levels"]
    error_type: str  # e.g., "conceptual_misunderstanding", "calculation_error", "misreading"
    root_cause: str  # Detailed explanation of what went wrong
    specific_mistake: str  # The exact reasoning that failed
    requires_knowledge: List[str]  # What knowledge/skills are needed
    difficulty_factors: List[str]  # What makes this question challenging
    domain: str  # Physics, Chemistry, Biology
    subdomain: str  # More specific subdomain


@dataclass
class QuestionTypeTopicGroup:
    """Group of failures with same type and topic"""
    question_type: str
    topics: List[str]
    failures: List[IndividualFailureAnalysis]
    common_error_patterns: List[str]
    shared_root_causes: List[str]
    required_knowledge: Set[str]
    key_difficulty_factors: List[str]
    domain: str


@dataclass
class TypeTopicEnhancement:
    """Enhancement specific to a question type and topic combination"""
    question_type: str
    topics: List[str]
    num_questions: int
    common_mistakes: List[str]
    key_warnings: List[str]
    verification_steps: List[str]
    topic_specific_guidance: str
    type_specific_approach: str
    enhanced_prompt_addition: str
    domain: str


@dataclass
class ComprehensiveEnhancement:
    """Complete enhancement with individual and grouped analyses"""
    strategy: str
    category: str  # High-level domain (Physics, Chemistry, Biology)
    individual_analyses: List[IndividualFailureAnalysis]
    type_topic_groups: List[QuestionTypeTopicGroup]
    type_topic_enhancements: List[TypeTopicEnhancement]
    enhanced_prompts: Dict[str, str]  # Keys: 'concise', 'specific', 'reasoning'


class UnifiedBenchmarkSystem:
    """Unified system with individual analysis first, then pattern analysis"""

    def __init__(self,
                 client: Together,
                 model: str = "moonshotai/Kimi-K2-Thinking",
                 base_prompt_key: str = "zero_shot_cot",
                 prompts_file: str = "prompts.json",
                 batch_size_individual: int = 20,
                 batch_size_pattern: int = 30,
                 max_questions_per_category: int = 50,
                 enhancement_types: List[str] = None,
                 use_stream: bool = False):
        """
        Initialize the unified benchmark system

        Args:
            client: Together AI client
            model: Model to use (e.g., moonshotai/Kimi-K2-Thinking, meta-llama/Llama-3.3-70B-Instruct-Turbo)
            base_prompt_key: Key of the base prompt in prompts.json
            prompts_file: Path to prompts JSON file
            batch_size_individual: Batch size for individual failure analysis
            batch_size_pattern: Batch size for pattern analysis
            max_questions_per_category: Maximum questions to analyze per category
            enhancement_types: List of enhancement types to generate ('concise', 'specific', 'reasoning')
            use_stream: Whether to use streaming API calls
        """
        self.client = client
        self.model = model
        self.base_prompt_key = base_prompt_key
        self.batch_size_individual = batch_size_individual
        self.batch_size_pattern = batch_size_pattern
        self.max_questions_per_category = max_questions_per_category
        self.enhancement_types = enhancement_types or ['concise', 'specific', 'reasoning']
        self.prompt_manager = PromptManager(prompts_file)
        self.use_stream = use_stream

        print(f"\n✓ Initialized Unified Three-Agent Benchmark System (GPQA)")
        print(f"  - Model: {model}")
        print(f"  - Enhancement types: {', '.join(self.enhancement_types)}")
        print(f"  - Batch size (individual): {batch_size_individual}")
        print(f"  - Batch size (pattern): {batch_size_pattern}")
        print(f"  - Max questions per category: {max_questions_per_category or 'all'}")
        print(f"  - Streaming: {use_stream}")

    def analyze_individual_failure(self, failure: Dict[str, Any], strategy: str, domain: str) -> IndividualFailureAnalysis:
        """Analyze a single GPQA failure to determine exact cause"""

        question = failure.get('question', '')
        correct_answer = failure.get('correct_answer', '')
        model_answer = failure.get('predicted_answer', '')
        correct_letter = failure.get('correct_letter', '')
        predicted_letter = failure.get('predicted_letter', '')
        subdomain = failure.get('subject', failure.get('subdomain', ''))
        answer_choices = failure.get('answer_choices', {})
        reasoning = failure.get('reasoning', '')

        # Format answer choices for context
        choices_text = ""
        if answer_choices:
            for letter in ['A', 'B', 'C', 'D']:
                if letter in answer_choices:
                    marker = "✓" if letter == correct_letter else ("✗" if letter == predicted_letter else " ")
                    choices_text += f"  {marker} {letter}. {answer_choices[letter][:200]}\n"

        prompt = f"""Analyze this failed GPQA (Graduate-Level Google-Proof Q&A) question. The model used "{strategy}" prompting strategy.

Domain: {domain}
Subdomain: {subdomain}

Question: {question[:2000]}

Answer Choices:
{choices_text}

Correct Answer: {correct_letter} - {correct_answer[:500]}
Model's Answer: {predicted_letter} - {model_answer[:500]}

Model's Reasoning (excerpt): {reasoning[:1000]}

Provide a comprehensive analysis in JSON format. IMPORTANT: Ensure all strings are properly escaped:
{{
    "question_type": "<type: factual/conceptual/calculation/application/analysis/comparison>",
    "topics": ["<specific scientific topic 1>", "<specific scientific topic 2>"],
    "error_type": "<type: conceptual_misunderstanding/calculation_error/misreading/incomplete_analysis/wrong_elimination/knowledge_gap>",
    "root_cause": "<detailed explanation of the fundamental reason for choosing {predicted_letter} instead of {correct_letter}>",
    "specific_mistake": "<the exact reasoning step or knowledge gap that led to the wrong answer>",
    "requires_knowledge": ["<scientific knowledge 1>", "<scientific knowledge 2>"],
    "difficulty_factors": ["<what makes this question challenging for AI>"]
}}

Focus on why the model chose the wrong answer in this multiple-choice context."""

        try:
            result = call_llm(
                client=self.client,
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=800,
                stream=self.use_stream
            )

            # Extract JSON from response
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                # Fix common escape issues
                json_str = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r'\\\\', json_str)
                analysis_data = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")

            return IndividualFailureAnalysis(
                question_id=failure.get('question_id', 'unknown'),
                question_text=question,
                correct_answer=correct_answer,
                model_answer=model_answer,
                correct_letter=correct_letter,
                predicted_letter=predicted_letter,
                question_type=analysis_data.get('question_type', 'unknown'),
                topics=analysis_data.get('topics', []),
                error_type=analysis_data.get('error_type', 'unknown'),
                root_cause=analysis_data.get('root_cause', ''),
                specific_mistake=analysis_data.get('specific_mistake', ''),
                requires_knowledge=analysis_data.get('requires_knowledge', []),
                difficulty_factors=analysis_data.get('difficulty_factors', []),
                domain=domain,
                subdomain=subdomain
            )

        except Exception as e:
            print(f"    ⚠️ Error analyzing individual failure: {e}")
            return IndividualFailureAnalysis(
                question_id=failure.get('question_id', 'unknown'),
                question_text=question,
                correct_answer=correct_answer,
                model_answer=model_answer,
                correct_letter=correct_letter,
                predicted_letter=predicted_letter,
                question_type='unknown',
                topics=['general'],
                error_type='unknown',
                root_cause='Analysis failed',
                specific_mistake='Could not determine',
                requires_knowledge=[],
                difficulty_factors=[],
                domain=domain,
                subdomain=subdomain
            )

    def analyze_individual_failures_batch(self, failures: List[Dict[str, Any]], strategy: str, domain: str) -> List[IndividualFailureAnalysis]:
        """Analyze multiple failures individually in batches"""

        print(f"\n{'='*70}")
        print(f"STEP 1: INDIVIDUAL FAILURE ANALYSIS")
        print(f"{'='*70}")

        # Limit number of questions only if max_questions_per_category is set
        if self.max_questions_per_category and len(failures) > self.max_questions_per_category:
            print(f"  Sampling {self.max_questions_per_category} from {len(failures)} questions...")
            step = len(failures) // self.max_questions_per_category
            failures = failures[::step][:self.max_questions_per_category]
            print(f"  Analyzing {len(failures)} questions...")
        else:
            print(f"  Analyzing ALL {len(failures)} questions...")

        all_analyses = []
        total_batches = (len(failures) + self.batch_size_individual - 1) // self.batch_size_individual

        for i in range(0, len(failures), self.batch_size_individual):
            batch = failures[i:i + self.batch_size_individual]
            batch_num = i // self.batch_size_individual + 1

            print(f"\n  Processing batch {batch_num}/{total_batches} ({len(batch)} questions)...")

            for j, failure in enumerate(batch):
                analysis = self.analyze_individual_failure(failure, strategy, domain)
                all_analyses.append(analysis)

                if (j + 1) % 5 == 0:
                    print(f"    Analyzed {j + 1}/{len(batch)} in batch...")

            # Rate limiting
            if i + self.batch_size_individual < len(failures):
                time.sleep(2)

        print(f"\n  ✓ Completed individual analysis of {len(all_analyses)} questions")

        # Summary statistics
        type_counts = Counter(a.question_type for a in all_analyses)
        error_counts = Counter(a.error_type for a in all_analyses)

        print(f"\n  📊 Analysis Summary:")
        print(f"     Question Types: {dict(type_counts)}")
        print(f"     Error Types: {dict(error_counts)}")

        return all_analyses

    def group_by_type_and_topic(self, analyses: List[IndividualFailureAnalysis], domain: str) -> List[QuestionTypeTopicGroup]:
        """Group failures by question type and topics"""

        print(f"\n{'='*70}")
        print(f"STEP 2: GROUPING BY TYPE AND TOPIC")
        print(f"{'='*70}")

        # Group by (type, topic combination)
        groups = defaultdict(list)

        for analysis in analyses:
            # Create a key from type and sorted topics
            topic_key = tuple(sorted(analysis.topics[:3]))  # Use top 3 topics
            key = (analysis.question_type, topic_key)
            groups[key].append(analysis)

        # Convert to QuestionTypeTopicGroup objects
        type_topic_groups = []

        for (q_type, topics), group_analyses in groups.items():
            if len(group_analyses) < 2:  # Skip groups with only 1 question
                continue

            # Aggregate patterns
            error_patterns = []
            root_causes = []
            required_knowledge = set()
            difficulty_factors = []

            for analysis in group_analyses:
                if analysis.specific_mistake:
                    error_patterns.append(analysis.specific_mistake)
                if analysis.root_cause:
                    root_causes.append(analysis.root_cause)
                required_knowledge.update(analysis.requires_knowledge)
                difficulty_factors.extend(analysis.difficulty_factors)

            group = QuestionTypeTopicGroup(
                question_type=q_type,
                topics=list(topics),
                failures=group_analyses,
                common_error_patterns=list(set(error_patterns)),
                shared_root_causes=list(set(root_causes)),
                required_knowledge=required_knowledge,
                key_difficulty_factors=list(set(difficulty_factors)),
                domain=domain
            )

            type_topic_groups.append(group)

        print(f"  ✓ Created {len(type_topic_groups)} type-topic groups")

        for group in sorted(type_topic_groups, key=lambda g: len(g.failures), reverse=True)[:10]:
            topics_str = ", ".join(group.topics[:2])
            print(f"     • {group.question_type} - [{topics_str}]: {len(group.failures)} questions")

        return type_topic_groups

    def analyze_patterns_in_group(self, group: QuestionTypeTopicGroup, strategy: str) -> TypeTopicEnhancement:
        """Analyze patterns within a type-topic group and create targeted enhancement"""

        topics_str = ", ".join(group.topics)
        failures_summary = "\n".join([
            f"- Q: {f.question_text[:100]}...\n  Correct: {f.correct_letter}, Predicted: {f.predicted_letter}\n  Error: {f.specific_mistake}\n  Cause: {f.root_cause}"
            for f in group.failures[:10]  # Use top 10 for analysis
        ])

        prompt = f"""Analyze these {len(group.failures)} failed GPQA questions that share:
- Domain: {group.domain}
- Question Type: {group.question_type}
- Topics: {topics_str}
- Prompting Strategy Used: {strategy}

Sample Failures:
{failures_summary}

Common Error Patterns: {', '.join(group.common_error_patterns[:5])}
Required Knowledge: {', '.join(list(group.required_knowledge)[:5])}
Difficulty Factors: {', '.join(group.key_difficulty_factors[:5])}

Create a targeted enhancement strategy for answering {group.domain} multiple-choice questions using "{strategy}" approach in JSON format:
{{
    "common_mistakes": ["<specific mistake pattern 1>", "<specific mistake pattern 2>"],
    "key_warnings": ["<critical warning for this topic 1>", "<critical warning 2>"],
    "verification_steps": ["<verification step 1>", "<verification step 2>"],
    "topic_specific_guidance": "<detailed guidance for {topics_str} questions>",
    "type_specific_approach": "<approach for {group.question_type} questions in {group.domain}>",
    "enhanced_prompt_addition": "<concise prompt addition (2-3 sentences) to help with this type/topic combination>"
}}

Make it specific to {group.question_type} questions about {topics_str} in {group.domain}."""

        try:
            result = call_llm(
                client=self.client,
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=1000,
                stream=self.use_stream
            )

            # Extract JSON
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                enhancement_data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found")

            return TypeTopicEnhancement(
                question_type=group.question_type,
                topics=group.topics,
                num_questions=len(group.failures),
                common_mistakes=enhancement_data.get('common_mistakes', []),
                key_warnings=enhancement_data.get('key_warnings', []),
                verification_steps=enhancement_data.get('verification_steps', []),
                topic_specific_guidance=enhancement_data.get('topic_specific_guidance', ''),
                type_specific_approach=enhancement_data.get('type_specific_approach', ''),
                enhanced_prompt_addition=enhancement_data.get('enhanced_prompt_addition', ''),
                domain=group.domain
            )

        except Exception as e:
            print(f"    ⚠️ Error analyzing group: {e}")
            return TypeTopicEnhancement(
                question_type=group.question_type,
                topics=group.topics,
                num_questions=len(group.failures),
                common_mistakes=[],
                key_warnings=[],
                verification_steps=[],
                topic_specific_guidance='',
                type_specific_approach='',
                enhanced_prompt_addition='',
                domain=group.domain
            )

    def analyze_patterns_all_groups(self, groups: List[QuestionTypeTopicGroup], strategy: str) -> List[TypeTopicEnhancement]:
        """Analyze patterns for all type-topic groups"""

        print(f"\n{'='*70}")
        print(f"STEP 3: PATTERN ANALYSIS FOR EACH TYPE-TOPIC GROUP")
        print(f"{'='*70}")
        print(f"Analyzing {len(groups)} groups...")

        enhancements = []

        for i, group in enumerate(groups):
            topics_str = ", ".join(group.topics[:2])
            print(f"\n  [{i+1}/{len(groups)}] Analyzing: {group.question_type} - [{topics_str}] ({len(group.failures)} questions)")

            enhancement = self.analyze_patterns_in_group(group, strategy)
            enhancements.append(enhancement)

            if (i + 1) % 5 == 0:
                time.sleep(2)  # Rate limiting

        print(f"\n  ✓ Completed pattern analysis for {len(enhancements)} groups")

        return enhancements

    def create_enhanced_prompts_all_types(self,
                                           base_prompt: str,
                                           enhancements: List[TypeTopicEnhancement],
                                           strategy: str,
                                           category: str) -> Dict[str, str]:
        """Create enhanced prompts for selected enhancement types"""

        print(f"\n{'='*70}")
        print(f"STEP 4: CREATING ENHANCED PROMPTS ({', '.join(self.enhancement_types).upper()})")
        print(f"{'='*70}")

        # Sort enhancements by number of questions (most common first)
        sorted_enhancements = sorted(enhancements, key=lambda e: e.num_questions, reverse=True)

        all_prompts = {}

        # 1. CONCISE ENHANCEMENT
        if 'concise' in self.enhancement_types:
            print("  Creating Concise Enhancement...")
            concise_text = f"\n\n## SPECIALIZED GUIDANCE FOR {category.upper()} ({strategy.upper()})\n\n"
            concise_text += "### Critical Warnings by Question Type:\n\n"

            for enh in sorted_enhancements[:8]:  # Top 8 type-topic combinations
                topics_str = "/".join(enh.topics[:2])
                concise_text += f"**{enh.question_type.title()} questions about {topics_str}** ({enh.num_questions} failures):\n"

                if enh.key_warnings:
                    concise_text += "⚠️  " + " | ".join(enh.key_warnings[:3]) + "\n"

                if enh.enhanced_prompt_addition:
                    concise_text += f"→ {enh.enhanced_prompt_addition}\n"

                concise_text += "\n"

            all_prompts['concise'] = base_prompt + concise_text

        # 2. SPECIFIC ENHANCEMENT
        if 'specific' in self.enhancement_types:
            print("  Creating Specific Enhancement...")
            specific_text = f"\n\n## SPECIALIZED GUIDANCE FOR {category.upper()} ({strategy.upper()})\n\n"
            specific_text += "### Detailed Guidance by Question Type and Topic:\n\n"

            for enh in sorted_enhancements[:10]:
                topics_str = " & ".join(enh.topics[:2])
                specific_text += f"**{enh.question_type.title()} - {topics_str}** ({enh.num_questions} failures):\n\n"

                if enh.common_mistakes:
                    specific_text += "Common Mistakes:\n"
                    for mistake in enh.common_mistakes[:3]:
                        specific_text += f"  • {mistake}\n"

                if enh.verification_steps:
                    specific_text += "\nVerification Steps:\n"
                    for step in enh.verification_steps[:4]:
                        specific_text += f"  ✓ {step}\n"

                if enh.type_specific_approach:
                    specific_text += f"\nApproach: {enh.type_specific_approach}\n"

                specific_text += "\n"

            all_prompts['specific'] = base_prompt + specific_text

        # 3. REASONING ENHANCEMENT
        if 'reasoning' in self.enhancement_types:
            print("  Creating Reasoning Enhancement...")
            reasoning_text = f"\n\n## SPECIALIZED GUIDANCE FOR {category.upper()} ({strategy.upper()})\n\n"
            reasoning_text += "### Key Considerations by Problem Type:\n\n"

            for enh in sorted_enhancements[:6]:
                topics_str = "/".join(enh.topics[:2])
                reasoning_text += f"• **{enh.question_type.title()} ({topics_str})**: {enh.enhanced_prompt_addition}\n"

            all_prompts['reasoning'] = base_prompt + reasoning_text

        print(f"  ✓ Created {len(all_prompts)} enhancement type(s) with {len(sorted_enhancements)} type-topic specific enhancements")

        return all_prompts

    def process_failed_questions(self,
                                 failures: List[Dict[str, Any]],
                                 strategy: str,
                                 category: str) -> ComprehensiveEnhancement:
        """Process failed questions through the complete pipeline"""

        # Step 1: Individual analysis
        individual_analyses = self.analyze_individual_failures_batch(failures, strategy, category)

        # Step 2: Group by type and topic
        type_topic_groups = self.group_by_type_and_topic(individual_analyses, category)

        # Step 3: Pattern analysis for each group
        type_topic_enhancements = self.analyze_patterns_all_groups(type_topic_groups, strategy)

        # Step 4: Create all three enhanced prompts
        base_prompt = self.prompt_manager.get_prompt_template(strategy)
        enhanced_prompts = self.create_enhanced_prompts_all_types(
            base_prompt,
            type_topic_enhancements,
            strategy,
            category
        )

        return ComprehensiveEnhancement(
            strategy=strategy,
            category=category,
            individual_analyses=individual_analyses,
            type_topic_groups=type_topic_groups,
            type_topic_enhancements=type_topic_enhancements,
            enhanced_prompts=enhanced_prompts
        )

    def save_results(self,
                     output_dir: Path,
                     enhancement: ComprehensiveEnhancement,
                     strategy: str,
                     category: str):
        """Save all analysis results including all three enhancement types"""

        # Create subdirectory for this strategy-category
        subdir = output_dir / f"{strategy}_{category}"
        subdir.mkdir(parents=True, exist_ok=True)

        # Save individual analyses
        individual_data = {
            "total_questions": len(enhancement.individual_analyses),
            "strategy": strategy,
            "category": category,
            "model": self.model,
            "analyses": [
                {
                    "question_id": a.question_id,
                    "question": a.question_text,
                    "correct_answer": a.correct_answer,
                    "model_answer": a.model_answer,
                    "correct_letter": a.correct_letter,
                    "predicted_letter": a.predicted_letter,
                    "question_type": a.question_type,
                    "topics": a.topics,
                    "error_type": a.error_type,
                    "root_cause": a.root_cause,
                    "specific_mistake": a.specific_mistake,
                    "requires_knowledge": a.requires_knowledge,
                    "difficulty_factors": a.difficulty_factors,
                    "domain": a.domain,
                    "subdomain": a.subdomain
                }
                for a in enhancement.individual_analyses
            ]
        }

        with open(subdir / "01_individual_analyses.json", 'w', encoding='utf-8') as f:
            json.dump(individual_data, f, indent=2, ensure_ascii=False)

        # Save type-topic groups
        groups_data = {
            "total_groups": len(enhancement.type_topic_groups),
            "strategy": strategy,
            "category": category,
            "model": self.model,
            "groups": [
                {
                    "question_type": g.question_type,
                    "topics": g.topics,
                    "num_questions": len(g.failures),
                    "common_error_patterns": g.common_error_patterns,
                    "shared_root_causes": g.shared_root_causes,
                    "required_knowledge": list(g.required_knowledge),
                    "key_difficulty_factors": g.key_difficulty_factors,
                    "domain": g.domain
                }
                for g in enhancement.type_topic_groups
            ]
        }

        with open(subdir / "02_type_topic_groups.json", 'w', encoding='utf-8') as f:
            json.dump(groups_data, f, indent=2, ensure_ascii=False)

        # Save enhancements
        enhancements_data = {
            "total_enhancements": len(enhancement.type_topic_enhancements),
            "strategy": strategy,
            "category": category,
            "model": self.model,
            "enhancements": [
                {
                    "question_type": e.question_type,
                    "topics": e.topics,
                    "num_questions": e.num_questions,
                    "common_mistakes": e.common_mistakes,
                    "key_warnings": e.key_warnings,
                    "verification_steps": e.verification_steps,
                    "topic_specific_guidance": e.topic_specific_guidance,
                    "type_specific_approach": e.type_specific_approach,
                    "enhanced_prompt_addition": e.enhanced_prompt_addition,
                    "domain": e.domain
                }
                for e in enhancement.type_topic_enhancements
            ]
        }

        with open(subdir / "03_type_topic_enhancements.json", 'w', encoding='utf-8') as f:
            json.dump(enhancements_data, f, indent=2, ensure_ascii=False)

        # Save all three enhanced prompts (text format)
        for enh_type, prompt_text in enhancement.enhanced_prompts.items():
            filename = f"04_enhanced_prompt_{enh_type}.txt"
            with open(subdir / filename, 'w', encoding='utf-8') as f:
                f.write(prompt_text)

        # Save all three enhanced prompts (JSON format)
        for enh_type, prompt_text in enhancement.enhanced_prompts.items():
            prompt_data = {
                "prompts": {
                    f"{enh_type}_enhanced_{strategy}_{category}": {
                        "name": f"{enh_type.title()} Enhanced Prompt ({strategy} - {category})",
                        "description": f"Enhanced with {enh_type} type-topic specific approach for {strategy} strategy on {category}",
                        "template": prompt_text,
                        "max_tokens": 3000,
                        "variables": ["question", "choices"],
                        "strategy": strategy,
                        "category": category,
                        "enhancement_type": enh_type,
                        "num_type_topic_groups": len(enhancement.type_topic_groups),
                        "total_questions_analyzed": len(enhancement.individual_analyses)
                    }
                },
                "model_settings": self.prompt_manager.model_settings,
                "metadata": {
                    "version": "2.0",
                    "description": f"Type-topic based enhanced prompts for {strategy} on {category}",
                    "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "enhancement_type": enh_type,
                    "strategy": strategy,
                    "category": category,
                    "model_used": self.model
                }
            }

            filename = f"05_enhanced_prompt_{enh_type}.json"
            with open(subdir / filename, 'w', encoding='utf-8') as f:
                json.dump(prompt_data, f, indent=2, ensure_ascii=False)

        print(f"\n  ✓ Saved all results to: {subdir}")
        print(f"     • Individual analyses")
        print(f"     • Type-topic groups")
        print(f"     • Type-topic enhancements")
        print(f"     • Enhanced prompts: {', '.join(enhancement.enhanced_prompts.keys())}")

    def load_gpqa_failures(self, input_dir: str) -> Dict[str, Dict[str, List[Dict]]]:
        """Load failed questions from GPQA benchmark output directory"""

        print(f"\n{'='*70}")
        print(f"LOADING GPQA FAILED QUESTIONS")
        print(f"{'='*70}")
        print(f"Directory: {input_dir}")

        results = {}

        # GPQA benchmark output structure: gpqa_results_YYYYMMDD_HHMMSS/
        # Files: Category_Difficulty_failures.json, all_results.json

        if not os.path.exists(input_dir):
            print(f"  ✗ Directory not found: {input_dir}")
            return {}

        # Find all failure JSON files
        json_files = glob.glob(os.path.join(input_dir, "*_failures.json"))

        if not json_files:
            # Try subdirectories
            json_files = glob.glob(os.path.join(input_dir, "**/*_failures.json"), recursive=True)

        if not json_files:
            print(f"  ✗ No failure files found")
            return {}

        print(f"  Found {len(json_files)} failure file(s)")

        # Group by category (high-level domain)
        # Default strategy since GPQA benchmark doesn't track strategy in output
        default_strategy = "zero_shot_cot"
        results[default_strategy] = {}

        for file_path in json_files:
            try:
                filename = os.path.basename(file_path)

                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                category = data.get('category', 'Unknown')
                failures = data.get('failures', [])

                if failures:
                    # Use category as key (Physics, Chemistry, Biology)
                    if category not in results[default_strategy]:
                        results[default_strategy][category] = []

                    results[default_strategy][category].extend(failures)
                    print(f"  ✓ {category}: {len(failures)} failures from {filename}")

            except Exception as e:
                print(f"  ✗ Error loading {file_path}: {e}")

        # Print summary
        total_failures = sum(
            len(failures)
            for categories in results.values()
            for failures in categories.values()
        )
        print(f"\n  Total failures loaded: {total_failures}")

        return results

    def process_benchmark_failures(self,
                                   input_dir: str,
                                   output_dir: str,
                                   strategies: Optional[List[str]] = None,
                                   categories: Optional[List[str]] = None,
                                   min_failures: int = 5) -> Dict[str, Dict[str, str]]:
        """Process ALL failed questions by strategy and category"""

        # Load organized failures
        organized_failures = self.load_gpqa_failures(input_dir)

        if not organized_failures:
            print("  ✗ No failures loaded!")
            return {}

        # Create output directory
        session_output = Path(output_dir) / f"enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session_output.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"UNIFIED THREE-AGENT TYPE-TOPIC ENHANCEMENT SYSTEM (GPQA)")
        print(f"Generating: Concise, Specific, and Reasoning enhancements")
        print(f"Model: {self.model}")
        print(f"{'='*70}")

        all_enhancements = {}

        # Count total combinations
        total_combinations = sum(
            1 for strategy, categories_dict in organized_failures.items()
            if not strategies or strategy in strategies
            for category, failures in categories_dict.items()
            if (not categories or category in categories) and len(failures) >= min_failures
        )

        current = 0

        # Process each strategy and category combination
        for strategy, categories_dict in organized_failures.items():
            if strategies and strategy not in strategies:
                continue

            for category, failures in categories_dict.items():
                if categories and category not in categories:
                    continue

                if len(failures) < min_failures:
                    print(f"  Skipping {strategy}/{category}: only {len(failures)} failures (min: {min_failures})")
                    continue

                current += 1

                print(f"\n{'='*70}")
                print(f"Processing [{current}/{total_combinations}]: {strategy} - {category}")
                if self.max_questions_per_category:
                    print(f"Processing up to {self.max_questions_per_category} of {len(failures)} failed questions")
                else:
                    print(f"Processing ALL {len(failures)} failed questions")
                print(f"{'='*70}")

                # Process through complete pipeline
                comprehensive_enhancement = self.process_failed_questions(
                    failures,
                    strategy,
                    category
                )

                # Save results
                self.save_results(
                    session_output,
                    comprehensive_enhancement,
                    strategy,
                    category
                )

                # Store all three enhancements
                key = f"{strategy}_{category}"
                all_enhancements[key] = comprehensive_enhancement.enhanced_prompts

        print(f"\n{'='*70}")
        print(f"✅ All strategies and categories processed!")
        print(f"📁 Results saved to: {session_output}")
        print(f"\n📊 Processing Summary:")
        total_questions = 0
        for strategy, categories_dict in organized_failures.items():
            if strategies and strategy not in strategies:
                continue
            for category, failures in categories_dict.items():
                if categories and category not in categories:
                    continue
                if len(failures) >= min_failures:
                    total_questions += len(failures)
                    print(f"   • {strategy} - {category}: {len(failures)} questions")
        if self.max_questions_per_category:
            print(f"\n   Total: {total_questions} questions (up to {self.max_questions_per_category} per category)")
        else:
            print(f"\n   Total: {total_questions} questions analyzed (ALL failed questions)")
        print(f"   Generated: {len(self.enhancement_types)} enhancement type(s) per combination ({', '.join(self.enhancement_types)})")
        print(f"   Workflow: Individual Analysis → Type/Topic Grouping → Pattern Analysis → Enhancements")
        print(f"   Model used: {self.model}")
        print(f"{'='*70}")

        return all_enhancements


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Unified Type-Topic Enhancement System for GPQA - Processes failed questions from benchmark_gpqa.py (Together AI API)',
        epilog='''
Examples:
  # Process failures using Kimi-K2-Thinking model (default)
  python gpqa_enhancement_together.py --input gpqa_results_20251224_120000

  # Process with Llama model
  python gpqa_enhancement_together.py --input gpqa_results_20251224_120000 --model meta-llama/Llama-3.3-70B-Instruct-Turbo

  # Use streaming mode
  python gpqa_enhancement_together.py --input gpqa_results_20251224_120000 --stream

  # Process with limited questions per category (faster)
  python gpqa_enhancement_together.py --input gpqa_results_20251224_120000 --max-questions 30

  # Generate only specific enhancement type
  python gpqa_enhancement_together.py --input gpqa_results_20251224_120000 --enhancement-types concise specific

  # Process only Physics failures
  python gpqa_enhancement_together.py --input gpqa_results_20251224_120000 --categories Physics

Available models on Together AI:
  - moonshotai/Kimi-K2-Thinking (default)
  - meta-llama/Llama-3.3-70B-Instruct-Turbo
  - meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo
  - mistralai/Mixtral-8x22B-Instruct-v0.1
  - Qwen/Qwen2.5-72B-Instruct-Turbo
  - deepseek-ai/DeepSeek-V3
  - google/gemma-2-27b-it
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input', default='gpqa_results_20251225_032236',
                        help='Path to GPQA results directory (from benchmark_gpqa.py)')
    parser.add_argument('--output-dir', type=str, default='gpqa_enhanced_prompts',
                        help='Output directory for enhanced prompts')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-72B-Instruct-Turbo',
                        help='Model to use (e.g., moonshotai/Kimi-K2-Thinking, meta-llama/Llama-3.3-70B-Instruct-Turbo)')
    parser.add_argument('--base-prompt-key', type=str, default='zero_shot',
                        help='Key of the base prompt to use from prompts.json')
    parser.add_argument('--prompts-file', type=str, default='prompts.json',
                        help='Path to the prompts JSON file')
    parser.add_argument('--batch-size-individual', type=int, default=20,
                        help='Batch size for individual analysis')
    parser.add_argument('--batch-size-pattern', type=int, default=30,
                        help='Batch size for pattern analysis')
    parser.add_argument('--max-questions', type=int, default=None,
                        help='Maximum questions to analyze per category (default: None = all)')
    parser.add_argument('--min-failures', type=int, default=2,
                        help='Minimum failures required to process a category (default: 2)')
    parser.add_argument('--strategies', nargs='+',
                        default=None,
                        help='Strategies to process (default: all found)')
    parser.add_argument('--categories', nargs='+',
                        help='Specific categories to process (Physics, Chemistry, Biology)')
    parser.add_argument('--enhancement-types', nargs='+',
                        choices=['concise', 'specific', 'reasoning'],
                        default=['concise', 'specific', 'reasoning'],
                        help='Enhancement types to generate (default: all three)')
    parser.add_argument('--api-key', type=str, default=None,
                        help='Together AI API key (or set TOGETHER_API_KEY env var)')
    parser.add_argument('--stream', action='store_true',
                        help='Use streaming API calls')

    args = parser.parse_args()

    # Get API key from argument or environment variable
    api_key = args.api_key or os.getenv('TOGETHER_API_KEY')

    if not api_key:
        print("❌ Error: Together AI API key not found!")
        print("   Set TOGETHER_API_KEY environment variable or use --api-key")
        exit(1)

    # Create client
    client = create_client(api_key=api_key)

    print("\n" + "=" * 70)
    print("UNIFIED THREE-AGENT TYPE-TOPIC ENHANCEMENT SYSTEM (GPQA)")
    print("Together AI Edition")
    print("=" * 70)
    print(f"Input: {args.input}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model: {args.model}")
    print(f"Enhancement types: {', '.join(args.enhancement_types)}")
    print(f"Max questions per category: {args.max_questions or 'all'}")
    print(f"Min failures to process: {args.min_failures}")
    print(f"Streaming: {args.stream}")

    if args.strategies:
        print(f"Strategies to process: {', '.join(args.strategies)}")
    if args.categories:
        print(f"Categories to process: {', '.join(args.categories)}")

    # Create and run system
    system = UnifiedBenchmarkSystem(
        client=client,
        model=args.model,
        base_prompt_key=args.base_prompt_key,
        prompts_file=args.prompts_file,
        batch_size_individual=args.batch_size_individual,
        batch_size_pattern=args.batch_size_pattern,
        max_questions_per_category=args.max_questions,
        enhancement_types=args.enhancement_types,
        use_stream=args.stream
    )

    # Process benchmark failures
    enhanced_prompts = system.process_benchmark_failures(
        input_dir=args.input,
        output_dir=args.output_dir,
        strategies=args.strategies,
        categories=args.categories,
        min_failures=args.min_failures
    )

    if enhanced_prompts:
        print("\n" + "=" * 50)
        print(f"✨ Type-topic based enhancement complete!")
        print(f"Generated enhancement types: {', '.join(args.enhancement_types)}")
        print(f"Model used: {args.model}")
        print("Individual analysis → Type/Topic grouping → Pattern analysis → Targeted enhancements")
        print("=" * 50)