"""Stop generation before the reward model's context fills; never trim a finished answer."""
from __future__ import annotations

from .io import digest


class ArmoRMBudget:
    def __init__(self, options, tokenizer=None):
        self.options = dict(options)
        if not self.options.get('revision'):
            raise ValueError('pin the ArmoRM budget tokenizer revision')
        self.limit = self.options['limit']
        self.reserve = self.options['reserve_tokens']
        if type(self.limit) is not int or type(self.reserve) is not int or not 0 < self.reserve < self.limit:
            raise ValueError('invalid ArmoRM generation budget')
        if tokenizer is None:
            from transformers import AutoTokenizer, AutoConfig
            config = AutoConfig.from_pretrained(self.options['model_id'], revision=self.options['revision'])
            if config.max_position_embeddings != self.limit:
                raise ValueError('ArmoRM generation limit differs from pinned scorer context')
            tokenizer = AutoTokenizer.from_pretrained(self.options['model_id'], revision=self.options['revision'])
        self.tok = tokenizer

    def count(self, query, response):
        tokens = self.tok.apply_chat_template([
            {'role': 'user', 'content': query}, {'role': 'assistant', 'content': response}],
            tokenize=True, return_dict=False)
        if not isinstance(tokens, list) or (tokens and not isinstance(tokens[0], int)):
            raise ValueError('ArmoRM budget requires a flat token-ID list, not a tokenizer dictionary')
        return len(tokens)

    def check_prompt(self, query):
        tokens = self.count(query, '')
        if tokens >= self.limit - self.reserve:
            raise ValueError(f'full prompt leaves no safe ArmoRM output room: {tokens}; no prompt truncation')
        return tokens

    def stop(self, query, raw):
        tokens = self.count(query, raw)
        if tokens > self.limit:
            raise ValueError(f'ArmoRM generation guard overrun: {tokens}>{self.limit}; reject this generation')
        return tokens >= self.limit - self.reserve

    def metadata(self, query, raw):
        from .collect import answer_text
        tokens = self.count(query, answer_text(raw, 'final_only'))
        if tokens > self.limit:
            raise ValueError(f'final answer exceeds ArmoRM budget: {tokens}>{self.limit}')
        return {'armorm_generation_budget': {**self.options, 'tokens': tokens,
                 'prompt_tokens': self.count(query, ''),
                 'chat_template_hash': digest(self.tok.chat_template)}}

    def validate_step_guard(self, producer):
        # A token may expand into many reward-tokenizer tokens. Reserve slack for
        # one producer token and boundary re-tokenization; verify the actual bound
        # and reject any engine step that nevertheless overruns it.
        largest = max(len(producer.decode([i], skip_special_tokens=False).encode('utf-8'))
                      for i in range(producer.vocab_size))
        if largest + 16 > self.reserve:
            raise ValueError(f'producer token can exceed ArmoRM guard: {largest} bytes')
        return largest
