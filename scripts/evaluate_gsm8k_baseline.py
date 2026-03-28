from vllm import LLM, SamplingParams
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from collections.abc import Callable
from typing import List
from pathlib import Path
import json

base = Path(__file__).parent.parent

def main():
    prompts = []
    answers = []

    with open(base / 'cs336_alignment' / 'prompts' / 'r1_zero.prompt', 'r') as f:
        base_prompt = f.read()

    with open(base / 'data' / 'gsm8k' / 'test.jsonl', 'r') as f:
        for line in f:
            example = json.loads(line)  # parses the line into a dictionary
            prompts.append(base_prompt.replace('{question}', example['question']))
            ground_truth = example['answer'].split('####')[-1].strip()
            answers.append(ground_truth)

    output_path = base / 'outputs' / 'gsm8k_baseline.jsonl'
    llm = LLM(model="Qwen/Qwen2.5-Math-1.5B", dtype='half')
    sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=1024, stop=['</answer>'], include_stop_str_in_output=True)
    evaluate_vllm(llm, r1_zero_reward_fn, prompts, answers, sampling_params, output_path)

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    answers: List[str],
    eval_sampling_params: SamplingParams,
    output_path: Path
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """

    results = []

    outputs = vllm_model.generate(prompts, eval_sampling_params)
    counts = {'unformatted': 0, 'incorrect': 0, 'correct': 0}

    for output, answer in zip(outputs, answers):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        rewards = reward_fn(generated_text, answer)

        if rewards['format_reward'] == 0.0:
            counts['unformatted'] += 1
        elif rewards['answer_reward'] == 0.0:
            counts['incorrect'] += 1
        else:
            counts['correct'] += 1

        results.append({'prompt': prompt, 'response': generated_text, 'ground_truth': answer, 'rewards': rewards})

    print("Statistics:")
    print(f"Unformatted: {counts['unformatted']}")
    print(f"Formatted but incorrect: {counts['incorrect']}")
    print(f"Formatted and correct: {counts['correct']}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')


if __name__ == '__main__':
    main()