"""Gemma 4 conversation adapter: model proposals are untrusted, tool registry is fixed.

Uses Google's official Transformers Gemma 4 multimodal chat template and
processor.parse_response when supported. No cloud inference API is used.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import Protocol
from .engine import Merlino

MODEL_DEFAULT = 'google/gemma-4-E2B-it'  # T4-friendly; E4B is optional
ALLOWED = {'get_training_context', 'propose_weekly_plan', 'explain_last_decision', 'request_human_handover'}

TOOL_SCHEMAS = [
    {'type':'function','function': {'name':'get_training_context',
     'description':'Read the synthetic runner graph summary, policy and state. Does not change data.',
     'parameters': {'type':'object','properties': {},'required': []}}},
    {'type':'function','function': {'name':'propose_weekly_plan',
     'description':'Propose a seven-day synthetic weekly training plan for independent SHACL review. Does not authorize or approve it.',
     'parameters': {'type':'object','properties': {
       'planned_km': {'type':'number','description':'Numeric total km, e.g. 42.5'},
       'style': {'type':'string','enum':['normal','consecutive_hard'],
                 'description':'Deterministic sample workout layout'}},
       'required': ['planned_km']}}},
    {'type':'function','function': {'name':'explain_last_decision',
     'description':'Get the last independent gate verdict and applicable policy revision.',
     'parameters': {'type':'object','properties': {},'required': []}}},
    {'type':'function','function': {'name':'request_human_handover',
     'description':'Ask the governed engine to hand control to a human; never impersonate the runner.',
     'parameters': {'type':'object','properties': {},'required': []}}},
]

SYSTEM = '''You are Merlino, a synthetic running-coach agent for an ontology-engineering workshop.
You must use the provided tools to inspect evidence and propose plans. Never claim to have
approved, scheduled or started a plan unless the gate result explicitly confirms it.
The ontology determines workout meanings; SHACL and separate human approval govern actions.
If the gate blocks a proposal, state the reason and invite a revised candidate.
Do not provide medical diagnoses or treat the fictional progression policy as safe advice.
Only synthetic runner Alex exists; no personal data may be transmitted to external AI APIs.
Human approval and changing ontology/policies are NOT available to you as tools.
'''


def _unquote_token(value: str):
    value = value.strip()
    if value.startswith('<|"|>') and value.endswith('<|"|>'):
        return value[5:-5]
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return value.strip("\"'")


def parse_gemma_calls(raw: str) -> list[dict]:
    """Parse Google's documented native tool-call token syntax (flat scalar args).

    Anything ambiguous is rejected; never eval model output or dynamically invoke names.
    """
    calls = []
    for name, body in re.findall(r'<\|tool_call>call:([a-zA-Z_][a-zA-Z_0-9]*)\{(.*?)\}(?:<tool_call\|>|<\|tool_call\|>)', raw, re.S):
        if name not in ALLOWED:
            calls.append({'name': name, 'arguments': {}, 'invalid': True})
            continue
        args = {}
        if body.strip():
            parts = re.findall(r'([a-zA-Z_][\w]*):(<\|"\|>.*?<\|"\|>|[^,}]+)(?:,|$)', body.strip())
            if not parts:
                calls.append({'name': name, 'arguments': {}, 'invalid': True})
                continue
            for key, value in parts:
                if key in args:
                    raise ValueError('Duplicate argument from model')
                args[key] = _unquote_token(value)
        calls.append({'name': name, 'arguments': args})
    return calls


def normalize_calls(parsed) -> list[dict]:
    if not isinstance(parsed, list):
        return []
    clean = []
    for c in parsed:
        if not isinstance(c, dict):
            continue
        body = c.get('function') or c
        if not isinstance(body, dict):
            continue
        name = body.get('name')
        params = body.get('arguments', {})
        if isinstance(params, str):
            try: params = json.loads(params)
            except ValueError: params = None
        clean.append({'name': name, 'arguments': params if isinstance(params, dict) else {},
                      'invalid': name not in ALLOWED or not isinstance(params, dict)})
    return clean


class TextModel(Protocol):
    def generate(self, messages: list[dict], tools: list[dict]) -> tuple[str, list[dict]]:
        """Return (assistant content, native normalized tool calls)."""


class Gemma4Transformers:
    """On-device/in-runtime inference; no inference service API calls.

    Load explicitly; never silently downgrade to a synthetic model.
    Hugging Face download during load requires consent/network access.
    """
    def __init__(self, model_id: str = MODEL_DEFAULT, quantize_4bit: bool = True,
                 max_new_tokens: int = 384):
        import torch
        from transformers import AutoProcessor, AutoModelForMultimodalLM
        if not torch.cuda.is_available():
            raise RuntimeError('A CUDA GPU is required for the Colab Gemma path. Use scripted mode or a suitable local runtime.')
        kwargs = {'device_map':'auto', 'dtype':'auto'}
        if quantize_4bit:
            from transformers import BitsAndBytesConfig
            kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.float16)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForMultimodalLM.from_pretrained(model_id, **kwargs)
        self.max_new_tokens = max_new_tokens
        self.model_id = model_id

    def generate(self, messages: list[dict], tools: list[dict]) -> tuple[str,list[dict]]:
        import torch
        template = self.processor.apply_chat_template(messages, tools=tools,
                        tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=template, return_tensors='pt').to(self.model.device)
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                            do_sample=False, pad_token_id=self.processor.tokenizer.eos_token_id)
        generation = self.processor.decode(output_ids[0][inputs['input_ids'].shape[-1]:],
                                           skip_special_tokens=False)
        if hasattr(self.processor, 'parse_response'):
            parsed = self.processor.parse_response(generation)
            result = normalize_calls(parsed.get('tool_calls', []))
            content = parsed.get('content') or ''
        else:
            result = parse_gemma_calls(generation)
            content = re.sub(r'<\|tool_call>.*?<tool_call\|><\|tool_response>', '', generation, flags=re.S)
        return str(content).strip(), result


@dataclass
class Turn:
    response: str
    tool_results: list[dict]
    model: str


class MerlinoAgent:
    def __init__(self, engine: Merlino, model: TextModel, model_name: str = 'Gemma 4'):
        self.engine, self.model, self.model_name = engine, model, model_name
        self.history: list[dict] = [{'role':'system','content': SYSTEM}]

    def _execute(self, call: dict) -> dict:
        name, args = call.get('name'), call.get('arguments')
        if call.get('invalid') or name not in ALLOWED or not isinstance(args, dict):
            return {'error':'UNKNOWN_TOOL: request rejected'}
        if name in ('get_training_context', 'explain_last_decision', 'request_human_handover') and args:
            return {'error':'Unexpected arguments'}
        if name == 'get_training_context':
            return self.engine.context()
        if name == 'explain_last_decision':
            return {'decision': self.engine.trace[-1].as_dict() if self.engine.trace else None,
                    'policy': self.engine.policy_version,
                    'meaning_revision': self.engine.meaning_revision}
        if name == 'request_human_handover':
            return self.engine.handover().as_dict()
        if name == 'propose_weekly_plan':
            if set(args) - {'planned_km', 'style'} or 'planned_km' not in args:
                return {'error':'Invalid proposal arguments'}
            km = args['planned_km']
            if isinstance(km, bool) or not isinstance(km, (int, float)):
                return {'error':'planned_km must be numeric'}
            return self.engine.propose(km, args.get('style', 'normal')).as_dict()
        return {'error':'Tool unavailable'}

    def ask(self, question: str, *, max_tool_calls: int = 2) -> Turn:
        if not isinstance(question, str) or not 1 <= len(question) <= 1500:
            raise ValueError('Workshop prompt must be 1–1500 characters')
        self.history.append({'role':'user','content': question})
        calls_done = []
        for turn in range(max_tool_calls + 1):
            content, calls = self.model.generate(self.history, TOOL_SCHEMAS)
            if not calls:
                answer = content or 'No actionable answer produced. Inspect the graph and try again.'
                self.history.append({'role':'assistant','content': answer})
                return Turn(answer, calls_done, self.model_name)
            if turn >= max_tool_calls:
                answer = 'Stopped: tool-call budget reached; no further actions executed.'
                self.history.append({'role':'assistant','content':answer})
                return Turn(answer, calls_done, self.model_name)
            # Execute AT MOST one tool request per generation, to prevent multi-tool race/approval confusion.
            if len(calls) != 1:
                result = {'error':'Exactly one tool call per step is required'}
                call = {'name':'rejected_tool_call','arguments':{}}
            else:
                call = calls[0]
                result = self._execute(call)
            calls_done.append({'name':call.get('name'), 'arguments':call.get('arguments'), 'result':result})
            # Google's Gemma 4 chat template expects tool_calls and tool_responses on assistant message.
            self.history.append({'role':'assistant',
                                 'tool_calls':[{'function': {'name':call['name'], 'arguments':call['arguments']}}],
                                 'tool_responses':[{'name':call['name'], 'response': result}]})
        raise AssertionError('unreachable')
