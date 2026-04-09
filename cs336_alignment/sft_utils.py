from transformers import PreTrainedTokenizer, PreTrainedModel
import torch


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizer
) -> dict[str, torch.Tensor]:
    prompt_tokens = []
    output_tokens = []
    concat_tokens = []
    for prompt, output in zip(prompt_strs, output_strs):
        tokenized_prompt = tokenizer.encode(prompt)
        tokenized_output = tokenizer.encode(output)
        tokenized_concat = tokenized_prompt + tokenized_output
        prompt_tokens.append(tokenized_prompt)
        output_tokens.append(tokenized_output)
        concat_tokens.append(tokenized_concat)

    maxlen = 0
    for tokens in concat_tokens:
        maxlen = max(maxlen, len(tokens))
    for i in range(len(concat_tokens)):
        strlen = len(concat_tokens[i])
        concat_tokens[i] = concat_tokens[i] + [tokenizer.pad_token_id] * (maxlen - strlen)
    input_ids = torch.tensor(concat_tokens)[:, :-1]
    labels = torch.tensor(concat_tokens)[:, 1:]
    response_mask = torch.zeros_like(labels)
    for i in range(len(concat_tokens)):
        prompt_len = len(prompt_tokens[i])
        output_len = len(output_tokens[i])
        response_mask[i, prompt_len - 1: prompt_len + output_len - 1] = 1

    result_dict = {
        'input_ids': input_ids,
        'labels': labels,
        'response_mask': response_mask
    }
    return result_dict


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    logits = logits - logits.max(dim=-1, keepdim=True).values
    exp_logits = torch.exp(logits)
    normalized_logits = exp_logits / exp_logits.sum(dim=-1, keepdim=True)
    log_logits = torch.log(normalized_logits)
    return -(log_logits * normalized_logits).sum(dim=-1)


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool=False
) -> dict[str, torch.Tensor]:
    logits = model(input_ids).logits
    logits = logits - logits.max(dim=-1, keepdim=True).values
    exp_logits = torch.exp(logits)
    normalized_logits = exp_logits / exp_logits.sum(dim=-1, keepdim=True)
    probs = normalized_logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    log_probs = torch.log(probs)
    return_dict = {'log_probs': log_probs}
    if return_token_entropy:
        token_entropy = compute_entropy(logits)
        return_dict['token_entropy'] = token_entropy
    return return_dict


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None=None,
) -> torch.Tensor:
    tensor_masked = tensor * mask
    if dim is not None:
        tensor_normalized = tensor_masked.sum(dim=dim) / normalize_constant
    else:
        tensor_normalized = tensor_masked.sum() / normalize_constant
    return tensor_normalized


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss = -0.5 * masked_normalize(policy_log_probs, response_mask, normalize_constant)
    loss = loss / gradient_accumulation_steps
    loss.backward()
    metadata = {}
    return loss, metadata


