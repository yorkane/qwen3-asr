# -*- coding: utf-8 -*-
"""Fast unit checks (no GPU, no model weights).

Covers pure logic of the ASR stack: prompt building, output parsing,
alignment tokenization, tuning-knob env parsing and router concurrency.
Modules that cannot be imported in the current environment are skipped.
"""

import asyncio
import os
import sys
import threading
import time
import unittest


def _import_qwen3_vllm():
    from app.services.asr import qwen3_vllm  # noqa: PLC0415

    return qwen3_vllm


class PromptAndParsingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.m = _import_qwen3_vllm()
        except Exception as exc:  # noqa: BLE001
            raise unittest.SkipTest(f"qwen3_vllm not importable: {exc}")

    def test_language_alias_normalization(self):
        m = self.m
        self.assertEqual(m._normalize_language_name("zh"), "Chinese")
        self.assertEqual(m._normalize_language_name("ZH-CN"), "Chinese")
        self.assertEqual(m._normalize_language_name("en-us"), "English")
        self.assertEqual(m._normalize_language_name("yue"), "Cantonese")
        self.assertEqual(m._normalize_language_name("  "), None)
        self.assertEqual(m._normalize_language_name(None), None)

    def test_build_chat_prompt_contains_placeholder(self):
        m = self.m
        prompt = m._build_chat_prompt(context="", language="zh")
        self.assertIn("im_start", prompt)
        # audio placeholder pad token is <|audio_pad|>-style; assert it is present
        self.assertIn("audio", prompt.lower())
        self.assertIn("user", prompt)
        self.assertIn("assistant", prompt)
        prompt_ctx = m._build_chat_prompt(context="hotword-a", language=None)
        self.assertIn("hotword-a", prompt_ctx)

    def test_parse_asr_output_with_language_prefix(self):
        m = self.m
        lang, text = m._parse_asr_output(
            "language Chinese<asr_text>hello world", None
        )
        self.assertEqual(lang, "Chinese")
        self.assertEqual(text, "hello world")
        lang2, text2 = m._parse_asr_output("plain text only", "en")
        self.assertEqual(lang2, "en")
        self.assertEqual(text2, "plain text only")

    def test_split_alignment_units_cjk_and_latin(self):
        m = self.m
        units = m._split_alignment_units("hello世界123！")
        self.assertIn("hello", units)
        self.assertIn("世", units)
        self.assertIn("界", units)
        self.assertIn("123", units)
        self.assertEqual(m._split_alignment_units(""), [])

    def test_build_alignment_prompt_pairs_timestamps(self):
        m = self.m
        body = m._build_alignment_prompt(["你", "好"])
        self.assertIn("audio", body.lower())
        # 2 alignment tokens -> one separator pair + one trailing pair = 4 tags
        self.assertGreaterEqual(body.count("timestamp"), 4)


class TuningKnobTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.m = _import_qwen3_vllm()
        except Exception as exc:  # noqa: BLE001
            raise unittest.SkipTest(f"qwen3_vllm not importable: {exc}")

    def test_resolve_optional_int_env(self):
        m = self.m
        os.environ["QA_TEST_KNOB"] = "42"
        try:
            self.assertEqual(m._resolve_optional_int_env("QA_TEST_KNOB"), 42)
        finally:
            os.environ.pop("QA_TEST_KNOB", None)
        os.environ["QA_TEST_KNOB"] = "not-a-number"
        try:
            self.assertIsNone(m._resolve_optional_int_env("QA_TEST_KNOB"))
        finally:
            os.environ.pop("QA_TEST_KNOB", None)
        self.assertIsNone(m._resolve_optional_int_env("QA_TEST_MISSING"))

    def test_max_in_flight_default_and_override(self):
        m = self.m
        os.environ["QWEN_VLLM_MAX_IN_FLIGHT"] = "7"
        try:
            self.assertEqual(m._resolve_max_in_flight(), 7)
        finally:
            os.environ.pop("QWEN_VLLM_MAX_IN_FLIGHT", None)
        self.assertEqual(m._resolve_max_in_flight(), 64)


class RouterConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    """Offline vLLM requests must overlap (async engine batches them)."""

    async def test_offline_requests_run_concurrently(self):
        try:
            from app.services.asr.engines import ASRFullResult
            from app.services.asr.runtime.router import (
                OfflineASRRequest,
                RuntimeFamily,
                RuntimeRouter,
            )
        except Exception as exc:  # noqa: BLE001
            raise unittest.SkipTest(f"router not importable: {exc}")

        class _StatefulEngine:
            def __init__(self):
                self._lock = threading.Lock()
                self.active = 0
                self.max_active = 0

            def transcribe_long_audio(self, *, audio_path, **_kwargs):
                with self._lock:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                time.sleep(0.05)
                with self._lock:
                    self.active -= 1
                return ASRFullResult(text=audio_path, segments=[], duration=0.0)

        engine = _StatefulEngine()
        router = RuntimeRouter()
        semaphore = asyncio.Semaphore(64)
        router._resolve_family = lambda _m: RuntimeFamily.QWEN_VLLM
        router._get_shared_engine = lambda _f, _m: (engine, semaphore)
        requests = [
            OfflineASRRequest(model_id="qwen3-asr-test", audio_path=f"req-{i}")
            for i in range(8)
        ]
        results = await asyncio.gather(*(router.run_offline(r) for r in requests))
        self.assertGreater(engine.max_active, 1, "requests were serialized")
        self.assertEqual(engine.active, 0)
        self.assertEqual(
            [r.text for r in results], [r.audio_path for r in requests]
        )


def run(verbosity: int = 1) -> bool:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in (PromptAndParsingTests, TuningKnobTests, RouterConcurrencyTests):
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    skipped = len(result.skipped)
    if skipped:
        print(f"[unit] skipped {skipped} test class(es) (missing optional deps)")
    return result.wasSuccessful()


if __name__ == "__main__":
    sys.exit(0 if run(verbosity=2) else 1)
