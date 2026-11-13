"""CPU-only tests for packing MLA weights without materializing full tables."""
import numpy as np
import pytest

from tinygrad import Tensor
from tinygrad.llm.model import Transformer, TransformerConfig
from tinygrad.llm.kernels.rockchip_mla import configure_weights
from tinygrad.llm.kernels.rockchip_mla_cpu import CPUMLAProjections


def model_with_tables(*, heads=4, nope=64, latent=64, value=64, seed=1701):
  config = TransformerConfig(1, heads * value, 512, heads, heads, 1e-5, 128, 96, 10000.0, 32, value,
                            max_context=128, q_lora_rank=32, kv_lora_rank=latent)
  model = Transformer(config)
  rng = np.random.default_rng(seed)
  block = model.blk[0]
  # Transpose square and rectangular backing arrays so the validation path sees
  # real lazy/non-contiguous source views before packing.
  key_base = Tensor(rng.normal(0, 0.03, (heads, nope, latent)).astype(np.float32), device="CPU").half()
  value_base = Tensor(rng.normal(0, 0.03, (heads, latent, value)).astype(np.float32), device="CPU").half()
  block.attn_k_b["weight"] = key_base.permute(0, 2, 1)
  block.attn_v_b["weight"] = value_base.permute(0, 2, 1)
  return model


def _bits(tensor):
  return tensor.numpy().view(np.uint16)


@pytest.mark.parametrize("heads,nope,latent,value", [(4, 64, 64, 64), (40, 64, 256, 64)])
def test_pack_once_keeps_source_uops_and_matches_eager_packed_tables(heads, nope, latent, value):
  candidate = model_with_tables(heads=heads, nope=nope, latent=latent, value=value)
  control = model_with_tables(heads=heads, nope=nope, latent=latent, value=value)
  candidate_block, control_block = candidate.blk[0], control.blk[0]
  candidate_sources = (candidate_block.attn_k_b["weight"], candidate_block.attn_v_b["weight"])
  identities = tuple(id(weight) for weight in candidate_sources)
  source_uops = tuple(weight.uop for weight in candidate_sources)

  assert configure_weights(candidate, materialize=False) == 2
  assert tuple(id(weight) for weight in candidate_sources) == identities
  assert tuple(weight.uop for weight in candidate_sources) == source_uops

  control_sources = (control_block.attn_k_b["weight"], control_block.attn_v_b["weight"])
  eager_uops = tuple(weight.uop for weight in control_sources)
  assert configure_weights(control) == 2
  assert all(weight.uop is not before for weight, before in zip(control_sources, eager_uops))
  source_values = tuple(_bits(weight).copy() for weight in control_sources)

  candidate_hook = CPUMLAProjections(*candidate_sources)
  control_hook = CPUMLAProjections(*control_sources)
  candidate_hook.prepare()
  control_hook.prepare()
  for actual, expected in zip(candidate_hook._packed(), control_hook._packed()):
    assert actual.numpy().tobytes() == expected.numpy().tobytes()

  rng = np.random.default_rng(1702)
  for role, depth, source_index in (("query", nope, 0), ("value", latent, 1)):
    hook = getattr(candidate_hook, role)
    weight = control_sources[source_index]
    for seed in (0, 1):
      x = Tensor(rng.normal(0, 0.2, (1, heads, 3 + seed, depth)).astype(np.float32), device="CPU").realize()
      expected = (x @ weight.transpose(-1, -2)).realize()
      np.testing.assert_array_equal(hook(x).numpy().view(np.uint32), expected.numpy().view(np.uint32))

    # Unsupported projection dtype takes the ordinary matmul fallback.
    fallback_x = Tensor(rng.normal(0, 0.2, (1, heads, 2, depth)).astype(np.float32), device="CPU").half().realize()
    fallback = (fallback_x @ control_sources[source_index].transpose(-1, -2)).realize()
    np.testing.assert_array_equal(hook(fallback_x).numpy().view(np.uint16), fallback.numpy().view(np.uint16))

  assert tuple(weight.uop for weight in candidate_sources) == source_uops
  for weight, before in zip(candidate_sources, source_values):
    np.testing.assert_array_equal(_bits(weight), before)


def test_pack_once_config_option_keeps_mla_sources_and_reports_prepared_geometry():
  from tinygrad.llm.kernels import rockchip

  model = model_with_tables()
  block = model.blk[0]
  sources = (block.attn_k_b["weight"], block.attn_v_b["weight"])
  identities = tuple(id(weight) for weight in sources)
  source_uops = tuple(weight.uop for weight in sources)
  placement = rockchip.configure(model, prefill="off", decode="off", cpu_pool=False,
                                 cpu_mla_weights=True, cpu_mla_projections=True, cpu_mla_pack_once=True)

  assert placement["cpu_mla_weights"] == 2
  assert placement["cpu_mla_projection_blocks"] == 1
  assert placement["cpu_mla_pack_once"] is True
  assert placement["cpu_mla_projection_packed_weight_bytes"] > 0
  assert tuple(id(weight) for weight in sources) == identities
  assert tuple(weight.uop for weight in sources) == source_uops
  assert block.cpu_mla_projections._sources() == sources


def test_pack_once_rejects_bad_types_and_missing_required_options():
  from tinygrad.llm.kernels import rockchip

  model = model_with_tables()
  with pytest.raises(ValueError, match="materialization selection"):
    configure_weights(model, materialize=1)
  with pytest.raises(ValueError, match="packed preparation selection"):
    rockchip.configure(model, cpu_mla_pack_once=1)
  with pytest.raises(ValueError, match="requires CPU weights and projections"):
    rockchip.configure(model, cpu_mla_pack_once=True, cpu_mla_projections=True)
  with pytest.raises(ValueError, match="requires CPU weights and projections"):
    rockchip.configure(model, cpu_mla_pack_once=True, cpu_mla_weights=True)


def test_materialize_false_validates_without_realize_or_replace(monkeypatch):
  model = model_with_tables()
  sources = (model.blk[0].attn_k_b["weight"], model.blk[0].attn_v_b["weight"])
  identities = tuple(id(weight) for weight in sources)
  source_uops = tuple(weight.uop for weight in sources)

  def unexpected(*args, **kwargs):
    raise AssertionError("materialize=False must not realize or replace source tables")

  monkeypatch.setattr(Tensor, "realize", unexpected)
  monkeypatch.setattr(Tensor, "replace", unexpected)
  assert configure_weights(model, materialize=False) == 2
  assert tuple(id(weight) for weight in sources) == identities
  assert tuple(weight.uop for weight in sources) == source_uops


def test_unsupported_pack_once_tables_fail_without_rockchip_device_lookup(monkeypatch):
  from tinygrad import Device
  from tinygrad.llm.kernels import rockchip

  model = model_with_tables()
  model.blk[0].attn_k_b["weight"] = model.blk[0].attn_k_b["weight"].float()

  class CpuOnlyDevice:
    def __getitem__(self, name):
      if name == "ROCKCHIP":
        raise AssertionError("Unsupported MLA tables reached NPU allocation")
      return Device[name]

  monkeypatch.setattr(rockchip, "Device", CpuOnlyDevice())
  with pytest.raises(ValueError, match="requires supported projection tables"):
    rockchip.configure(model, prefill="off", decode="off", fuse_projections=False, cpu_pool=False,
                       cpu_mla_weights=True, cpu_mla_projections=True, cpu_mla_pack_once=True)
  assert not hasattr(model.blk[0], "cpu_mla_projections")
