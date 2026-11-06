"""Regression for Transformers parse_response(prefix=..., tools=...) without GPU."""
from contextlib import nullcontext
from types import SimpleNamespace
import sys

from merlino.gemma_agent import Gemma4Transformers


def test_gemma_parser_receives_real_prompt_prefix_and_tool_schema(monkeypatch):
    class FakeInputs(dict):
        def to(self, device):
            assert device == 'cpu'
            return self

    class FakeProcessor:
        def __init__(self):
            self.tokenizer = SimpleNamespace(eos_token_id=99)
            self.parser_calls = []
        def apply_chat_template(self, messages, tools, tokenize, add_generation_prompt):
            assert messages[0]['role'] == 'user'
            assert tokenize is False and add_generation_prompt is True
            return '<chat><user>Alex asks for a running plan</user><assistant>'
        def __call__(self, text, return_tensors):
            assert text.startswith('<chat>') and return_tensors == 'pt'
            return FakeInputs(input_ids=SimpleNamespace(shape=[1, 3]))
        def decode(self, ids, skip_special_tokens):
            assert ids == [4, 5] and skip_special_tokens is False
            return '<|tool_call>call:propose_weekly_plan{planned_km:34.6}<|tool_call|>'
        def parse_response(self, response, *, prefix, tools):
            self.parser_calls.append((response, prefix, tools))
            assert '<assistant>' in prefix
            return {'content': '', 'tool_calls': [
                {'function': {'name': 'propose_weekly_plan',
                              'arguments': {'planned_km': 34.6}}}]}

    class FakeModel:
        device = 'cpu'
        def generate(self, **kwargs):
            assert kwargs['do_sample'] is False
            return [[1, 2, 3, 4, 5]]

    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(inference_mode=nullcontext))
    adapter = object.__new__(Gemma4Transformers)
    adapter.processor = FakeProcessor()
    adapter.model = FakeModel()
    adapter.max_new_tokens = 64
    tools = [{'type': 'function', 'function': {'name': 'propose_weekly_plan'}}]
    answer, calls = adapter.generate([{'role': 'user', 'content': 'Plan'}], tools)
    assert answer == ''
    assert calls == [{'name': 'propose_weekly_plan',
                      'arguments': {'planned_km': 34.6}, 'invalid': False}]
    assert len(adapter.processor.parser_calls) == 1
    assert adapter.processor.parser_calls[0][2] is tools
