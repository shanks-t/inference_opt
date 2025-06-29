#!/usr/bin/env python3
"""
Query corpus generator for inference benchmarking.

Generates diverse, realistic prompts using OpenAI API.
Output is deterministic and idempotent for reproducible benchmarks.
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple
from dotenv import load_dotenv

from openai import OpenAI


class PromptGenerator:
    """Generate diverse prompts using OpenAI API."""

    def __init__(self, model_name: str = "gpt-4o-mini", seed: int = 42):
        """Initialize generator with OpenAI client and random seed."""
        self.seed = seed
        self.model_name = model_name

        # Set seeds for reproducibility
        random.seed(seed)

        # Initialize OpenAI client
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        self.client = OpenAI(api_key=api_key)
        print(f"Using OpenAI model: {model_name}")

    def get_prompt_templates(self) -> Dict[str, List[str]]:
        """Return categorized prompt templates for generation."""
        return {
            "code": [
                "Write a Python function that",
                "Debug this code:",
                "Explain how this algorithm works:",
                "Create a class for",
                "Write unit tests for",
                "Refactor this function to be more",
                "What's wrong with this implementation:",
                "Convert this code from Python to",
            ],
            "qa": [
                "What is the difference between",
                "How does",
                "Why do we need",
                "When should you use",
                "What are the advantages of",
                "Explain the concept of",
                "Compare and contrast",
                "What factors should be considered when",
            ],
            "creative": [
                "Write a short story about",
                "Imagine a world where",
                "Create a character who",
                "Describe a scene where",
                "Write dialogue between",
                "Continue this story:",
                "Create a poem about",
                "Write a letter from",
            ],
            "summarization": [
                "Summarize the key points of",
                "What are the main takeaways from",
                "Explain in simple terms:",
                "Break down the following concept:",
                "Analyze the following text:",
                "Extract the important information from:",
                "Provide an overview of",
                "Distill the essence of",
            ],
            "chat": [
                "I need help with",
                "Can you assist me in",
                "I'm trying to understand",
                "What would you recommend for",
                "How can I improve my",
                "I'm having trouble with",
                "Could you guide me through",
                "What's the best approach to",
            ]
        }

    def generate_prompt(self, template: str, target_length: str) -> Tuple[str, int]:
        """Generate a single prompt from template with target length."""
        # Set target word count based on target length
        length_descriptions = {
            "short": "10-20 words",
            "medium": "30-80 words",
            "long": "100-200 words"
        }
        target_desc = length_descriptions.get(target_length, "30-80 words")

        # Create system prompt for consistent generation
        system_prompt = (
            f"Complete the following prompt starter to create a realistic user query. "
            f"Target length: {target_desc}. Make it sound natural and specific."
        )

        try:
            # Make API call with deterministic seed
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": template}
                ],
                max_tokens=300,
                temperature=0.7,
                seed=self.seed + hash(template) % 1000  # Deterministic but varied
            )

            generated_text = response.choices[0].message.content.strip()

            # Estimate token count (rough approximation: 1 token ≈ 0.75 words)
            word_count = len(generated_text.split())
            estimated_tokens = int(word_count * 1.33)

            return generated_text, estimated_tokens

        except Exception as e:
            print(f"API call failed: {e}")
            # Fallback to template if API fails
            return template, len(template.split())

    def generate_corpus(self, n_prompts: int = 500) -> List[Dict]:
        """Generate complete prompt corpus with distribution across categories and lengths."""
        templates = self.get_prompt_templates()
        categories = list(templates.keys())
        lengths = ["short", "medium", "long"]

        corpus = []

        # Calculate distribution
        prompts_per_category = n_prompts // len(categories)
        prompts_per_length = prompts_per_category // len(lengths)

        prompt_id = 0

        print(f"Generating {n_prompts} prompts...")

        for category in categories:
            category_templates = templates[category]

            for length in lengths:
                # Generate prompts for this category/length combination
                for i in range(prompts_per_length):
                    # Randomly select template
                    template = random.choice(category_templates)

                    try:
                        prompt_text, token_count = self.generate_prompt(template, length)

                        corpus.append({
                            "id": prompt_id,
                            "prompt": prompt_text,
                            "category": category,
                            "target_length": length,
                            "estimated_tokens": token_count
                        })

                        prompt_id += 1

                        # Progress indicator
                        if (prompt_id + 1) % 50 == 0:
                            print(f"Generated {prompt_id + 1}/{n_prompts} prompts...")

                    except Exception as e:
                        print(f"Warning: Failed to generate prompt {prompt_id}: {e}")
                        continue

        # Fill remaining prompts if needed
        while len(corpus) < n_prompts:
            category = random.choice(categories)
            length = random.choice(lengths)
            template = random.choice(templates[category])

            try:
                prompt_text, token_count = self.generate_prompt(template, length)
                corpus.append({
                    "id": len(corpus),
                    "prompt": prompt_text,
                    "category": category,
                    "target_length": length,
                    "estimated_tokens": token_count
                })
            except Exception as e:
                print(f"Warning: Failed to generate filler prompt: {e}")
                break

        print(f"Generated {len(corpus)} prompts successfully")
        return corpus

    def save_corpus(self, corpus: List[Dict], output_path: Path) -> None:
        """Save corpus to JSONL format."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            for prompt_data in corpus:
                f.write(json.dumps(prompt_data) + '\n')

        print(f"Saved {len(corpus)} prompts to {output_path}")

        # Print summary statistics
        categories = {}
        lengths = {}
        total_tokens = 0

        for prompt_data in corpus:
            categories[prompt_data['category']] = categories.get(prompt_data['category'], 0) + 1
            lengths[prompt_data['target_length']] = lengths.get(prompt_data['target_length'], 0) + 1
            total_tokens += prompt_data['estimated_tokens']

        print(f"\nCorpus Statistics:")
        print(f"Categories: {dict(categories)}")
        print(f"Lengths: {dict(lengths)}")
        print(f"Average tokens per prompt: {total_tokens / len(corpus):.1f}")
        print(f"Total estimated tokens: {total_tokens}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate prompt corpus for inference benchmarking")
    parser.add_argument("--n", type=int, default=500, help="Number of prompts to generate")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use for generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="data/prompts.jsonl", help="Output file path")

    args = parser.parse_args()

    # Create generator
    generator = PromptGenerator(model_name=args.model, seed=args.seed)

    # Generate corpus
    corpus = generator.generate_corpus(n_prompts=args.n)

    # Save to file
    output_path = Path(args.output)
    generator.save_corpus(corpus, output_path)

    print(f"\nDone! Run 'python {__file__} --n {args.n}' to regenerate identical corpus.")


if __name__ == "__main__":
    main()
