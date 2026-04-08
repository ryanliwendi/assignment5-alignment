from transformers import PreTrainedTokenizer
import torch


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizer
) -> dict[str, torch.Tensor]:
    combined_str = []
    for i in range(len(prompt_strs)):
        combined_str.append(prompt_strs[i] + output_strs[i])
    encoding = tokenizer(combined_str, padding=True, return_tensors='pt')
    input_ids = encoding['input_ids'][:, :-1]
    labels = encoding['input_ids'][:, 1::]
    response_mask = torch.zeros_like(labels)
    for i in range(len(output_strs)):
        output_len = tokenizer(output_strs[i], return_tensors='pt')['input_ids'].shape[-1]
        response_mask[i, -output_len::] = 1
    data = {'input_ids': input_ids, 'labels': labels, 'response_mask': response_mask}
    return data





