#!/usr/bin/env python3
"""CPU-only tests for quality-runner configuration and provenance wiring."""
import argparse
import ast
import copy
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import Mock

HERE = Path(__file__).resolve().parents[2] / "extra" / "rockchip"
HELPER_PATH = HERE / "quality_runner_options.py"
SPEC = importlib.util.spec_from_file_location("staged_quality_runner_options", HELPER_PATH)
OPTIONS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OPTIONS)


class ProjectionOptionScaffolding(unittest.TestCase):
  def parse(self, argv):
    parser = argparse.ArgumentParser()
    OPTIONS.add_projection_options(parser)
    return parser.parse_args(argv)

  def test_defaults_preserve_previous_configuration(self):
    args = self.parse([])
    self.assertEqual(OPTIONS.projection_options(args), {
      "share_projection_weights": False, "cpu_prefill_pool": False, "cpu_recurrent_scan": False})
    self.assertIsNone(OPTIONS.projection_option_error(mode="candidate", precision="int8", device="CPU",
      checkpoint_is_text=True, **vars(args)))

  def test_flags_forward_to_configure_kwargs(self):
    args = self.parse(["--share-projection-weights", "--cpu-prefill-pool"])
    configure = Mock(return_value={"share_projection_weights": True, "cpu_prefill_pool": True})
    actual = configure(**OPTIONS.projection_options(args))
    configure.assert_called_once_with(share_projection_weights=True, cpu_prefill_pool=True, cpu_recurrent_scan=False)
    self.assertTrue(actual["share_projection_weights"] and actual["cpu_prefill_pool"])

  def test_incompatible_candidate_combinations_are_rejected(self):
    combinations = [
      {"mode":"stock", "precision":"int8", "device":"CPU", "checkpoint_is_text":True},
      {"mode":"candidate", "precision":"fp16", "device":"CPU", "checkpoint_is_text":True},
      {"mode":"candidate", "precision":"int8", "device":"CL", "checkpoint_is_text":True},
      {"mode":"candidate", "precision":"int8", "device":"CPU", "checkpoint_is_text":False},
    ]
    for flags in (["--share-projection-weights"], ["--cpu-prefill-pool"], ["--cpu-recurrent-scan"]):
      args = self.parse(flags)
      for case in combinations:
        with self.subTest(flags=flags, case=case):
          self.assertIsNotNone(OPTIONS.projection_option_error(**case, **vars(args)))

  def test_quality_candidate_forwards_helper_options_into_backend_configure(self):
    tree = ast.parse((HERE / "quality_long_context.py").read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "configure"]
    self.assertTrue(any(any(k.arg is None and isinstance(k.value, ast.Call) and isinstance(k.value.func, ast.Name)
      and k.value.func.id == "projection_options" for k in call.keywords) for call in calls))

  def test_actual_cpu_configuration_forwards_all_requested_options(self):
    tree=ast.parse((HERE / 'quality_long_context.py').read_text())
    function=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='cpu_configuration')
    args=self.parse(['--share-projection-weights','--cpu-prefill-pool','--cpu-recurrent-scan'])
    args.precision='int8'
    args.cpu_packed_attention=args.cpu_fp16_qk=args.cpu_int8_qk=False
    configure=Mock(return_value={})
    scope={'args':args,'configure':configure,'input_scales':None,'projection_options':OPTIONS.projection_options}
    module=ast.Module(body=[copy.deepcopy(function)],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),'actual_quality_cpu_configuration','exec'),scope)
    marker=object()
    scope['cpu_configuration'](marker)
    self.assertEqual(configure.call_args.args,(marker,))
    for key in ('share_projection_weights','cpu_prefill_pool','cpu_recurrent_scan','cpu_attention','native_replay'):
      self.assertIs(configure.call_args.kwargs[key],True)
    self.assertEqual((configure.call_args.kwargs['prefill'],configure.call_args.kwargs['decode']),('on','on'))

  def test_run_source_hashes_include_harness_and_helper(self):
    tree = ast.parse((HERE / "quality_long_context.py").read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "quality_runner_hashes")
    digest=Mock(return_value='a'*64)
    scope={'Path':Path,'__file__':str(HERE/'quality_long_context.py'),'digest':digest}
    module=ast.Module(body=[copy.deepcopy(function)],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),'actual_quality_harness_hashes','exec'),scope)
    hashes=scope['quality_runner_hashes']()
    self.assertEqual(set(hashes),{'extra/rockchip/'+name for name in
      ('quality_long_context.py','quality_inference.py','quality_runner_options.py')})
    self.assertEqual(digest.call_count,3)
    self.assertTrue(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "update"
      and any(isinstance(a, ast.Call) and isinstance(a.func, ast.Name) and a.func.id == "quality_runner_hashes" for a in n.args)
      for n in ast.walk(tree)))


if __name__ == "__main__": unittest.main()
