from transformers import PreTrainedTokenizer
import torch


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizer
) -> dict[str, torch.Tensor]:
    '''

    Args:
        prompt_strs: List of prompt strings.
        output_strs: List of output strings.
        tokenizer: Tokenizer to use for tokenization.

    Returns:
        dict[str, torch.Tensor] Let prompt_and_output_lens be a list containing the lengths of
        the tokenized prompt and output strings. Then the returned dictionary should have the
        following keys:
        input_ids torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
        the tokenized prompt and output strings, with the final token sliced off.
        labels torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
        shifted input ids, i.e., the input ids without the first token.
        response_mask torch.Tensor of shape (batch_size, max(prompt_and_output_lens) -
        1): a mask on the response tokens in the labels.
    '''
    combined_str = []
    for i in range(len(prompt_strs)):
        combined_str.append((prompt_strs[i] + output_strs[i]))
    encoding = tokenizer(combined_str, padding=True, return_tensors='pt')
    input_ids = encoding['input_ids'][:, :-1]
    labels = encoding['input_ids'][:, 1::]
    response_mask = torch.zeros_like(labels)
    for i in range(len(output_strs)):
        output_len = tokenizer(output_strs[i], return_tensors='pt')['input_ids'].shape[-1]
        response_mask[i, -output_len::] = 1
    data = {'input_ids': input_ids, 'labels': labels, 'response_mask': response_mask}
    return data


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    logits = logits - logits.max(dim=-1, keepdim=True).values
    exp_logits = torch.exp(logits)
    normalized_logits = exp_logits / exp_logits.sum(dim=-1, keepdim=True)
    log_logits = torch.log(normalized_logits)
    return (log_logits * normalized_logits).sum(dim=-1)






