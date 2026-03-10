import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PIPELINE_DIR = Path(__file__).resolve().parent.parent / "eval" / "alpacaeval"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import evaluate_model
import pipeline_lib
import run_pipeline


class _TokenizerWithoutChatTemplate:
    bos_token = "<|begin_of_text|>"
    chat_template = None

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.last_messages = messages
        self.last_tokenize = tokenize
        self.last_add_generation_prompt = add_generation_prompt
        content = messages[0]["content"]
        if "begin_of_text" in (self.chat_template or ""):
            prefix = "<|begin_of_text|>"
        else:
            prefix = ""
        return f"{prefix}<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


class AlpacaEvalPipelineTest(unittest.TestCase):
    def test_load_preset_resolves_required_keys(self):
        for preset_name in (
            "llama3_base_simpo",
            "llama3_instruct_simpo",
            "repo_llama3",
            "repo_qwen3",
        ):
            preset = pipeline_lib.load_preset(preset_name)
            self.assertIn("model_name", preset)
            self.assertIn("pretty_name", preset)
            self.assertIn("prompt_template_text", preset)
            self.assertIn("use_custom_chat_template", preset)
            self.assertIn("generation", preset)

    def test_render_prompt_single_bos_keeps_exactly_one_bos(self):
        preset = pipeline_lib.load_preset("llama3_base_simpo")
        tokenizer = _TokenizerWithoutChatTemplate()
        prompt = pipeline_lib.render_prompt(
            template_text=preset["prompt_template_text"],
            use_custom_chat_template=True,
            row={"instruction": "Say hello"},
            bos_mode="single",
            tokenizer=tokenizer,
        )
        self.assertEqual(prompt.count("<|begin_of_text|>"), 1)
        self.assertIn("Say hello", prompt)
        self.assertEqual(tokenizer.last_messages, [{"role": "user", "content": "Say hello"}])
        self.assertFalse(tokenizer.last_tokenize)
        self.assertTrue(tokenizer.last_add_generation_prompt)

    def test_render_prompt_none_strips_bos(self):
        template_path = PIPELINE_DIR / "templates" / "llama3-nobos.txt"
        tokenizer = _TokenizerWithoutChatTemplate()
        prompt = pipeline_lib.render_prompt(
            template_text=template_path.read_text(encoding="utf-8"),
            use_custom_chat_template=True,
            row={"instruction": "Say hello"},
            bos_mode="none",
            tokenizer=tokenizer,
        )
        self.assertNotIn("<|begin_of_text|>", prompt)
        self.assertIn("Say hello", prompt)

    def test_render_prompt_can_use_model_original_template(self):
        tokenizer = _TokenizerWithoutChatTemplate()
        tokenizer.chat_template = "native-template"
        prompt = pipeline_lib.render_prompt(
            template_text="custom-template",
            use_custom_chat_template=False,
            row={"instruction": "Say hello"},
            bos_mode="tokenizer",
            tokenizer=tokenizer,
        )
        self.assertEqual(tokenizer.chat_template, "native-template")
        self.assertIn("Say hello", prompt)

    def test_serialize_model_outputs_sanitizes_generator(self):
        outputs = pipeline_lib.serialize_model_outputs(
            [{"instruction": "Question", "input": ""}],
            ["Answer"],
            generator_name="foo/bar\\baz",
        )
        self.assertEqual(outputs[0]["generator"], "foo_bar_baz")
        self.assertEqual(outputs[0]["instruction"], "Question")
        self.assertEqual(outputs[0]["output"], "Answer")

    def test_evaluate_outputs_uses_run_dir_outputs_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            outputs_dir = run_dir / "outputs"
            outputs_dir.mkdir()
            outputs_file = outputs_dir / "model_outputs.json"
            outputs_file.write_text(
                json.dumps([{"instruction": "Question", "output": "Answer", "generator": "demo"}]),
                encoding="utf-8",
            )

            fake_module = mock.Mock()
            fake_module.evaluate = mock.Mock()

            with mock.patch.object(evaluate_model, "preflight_evaluation_env"), \
                mock.patch.object(evaluate_model, "_resolve_reference_outputs", return_value=str(outputs_file)), \
                mock.patch.object(evaluate_model, "_load_and_sanitize_outputs", return_value="frame"), \
                mock.patch.object(evaluate_model, "_load_alpaca_eval_main", return_value=fake_module), \
                mock.patch.object(evaluate_model, "_run"):
                evaluate_model.evaluate_outputs(
                    model_name="demo/model",
                    output_dir=str(run_dir),
                    run_dir=str(run_dir),
                    skip_generation=True,
                    skip_eval=False,
                    skip_analysis=True,
                )

            fake_module.evaluate.assert_called_once()
            kwargs = fake_module.evaluate.call_args.kwargs
            self.assertEqual(kwargs["model_outputs"], "frame")
            self.assertEqual(kwargs["reference_outputs"], "frame")
            self.assertEqual(Path(kwargs["output_path"]).resolve(), (run_dir / "results").resolve())

    def test_run_pipeline_smoke_with_mocked_generation_and_judge(self):
        fake_dataset = [
            {"instruction": "Question 1", "input": ""},
            {"instruction": "Question 2", "input": ""},
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            with mock.patch.object(pipeline_lib, "load_alpacaeval_dataset", return_value=fake_dataset):
                run_pipeline.main(
                    [
                        "prepare",
                        "--preset",
                        "repo_llama3",
                        "--run_name",
                        "smoke",
                        "--output_root",
                        str(output_root),
                    ]
                )

            run_dir = output_root / "smoke"

            with mock.patch.object(
                pipeline_lib,
                "generate_texts_from_prompts",
                return_value=["Answer 1", "Answer 2"],
            ), mock.patch.object(
                pipeline_lib.AutoTokenizer,
                "from_pretrained",
                return_value=_TokenizerWithoutChatTemplate(),
            ):
                run_pipeline.main(["generate", "--run_dir", str(run_dir), "--device", "cpu", "--batch_size", "1"])

            outputs = json.loads((run_dir / "outputs" / "model_outputs.json").read_text(encoding="utf-8"))
            self.assertEqual(len(outputs), 2)
            self.assertEqual(outputs[0]["output"], "Answer 1")

            with mock.patch.object(run_pipeline, "evaluate_outputs") as evaluate_outputs_mock:
                run_pipeline.main(["judge", "--run_dir", str(run_dir), "--skip_analysis"])
            evaluate_outputs_mock.assert_called_once()

            leaderboard_path = run_dir / "results" / "leaderboard.csv"
            leaderboard_path.write_text(
                "name,win_rate,length_controlled_winrate,avg_length\nrepo-llama3,0.1,0.2,42\n",
                encoding="utf-8",
            )
            (run_dir / "results" / "annotations.json").write_text("[]", encoding="utf-8")
            report = pipeline_lib.build_report(run_dir)
            self.assertEqual(report["run_name"], "smoke")
            self.assertEqual(report["leaderboard"]["name"], "repo-llama3")


if __name__ == "__main__":
    unittest.main()
