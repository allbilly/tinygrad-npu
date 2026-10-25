"""Allocation-free public configuration/ownership tests for selective GDN packing."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from tinygrad import Tensor, dtypes, nn
from tinygrad.llm.model import GatedDeltaNetBlock

from tinygrad.llm.kernels import mali


class Packed:
  def __init__(self,weight):
    self.shape=((weight.shape[0]+31)//32,weight.shape[1],32)
    self.dtype,self.device=weight.dtype,weight.device
  def realize(self):return self
  def nbytes(self):return self.shape[0]*self.shape[1]*self.shape[2]*self.dtype.itemsize


def linear(columns,depth,*,dtype=dtypes.half,device='CL'):
  source=nn.Linear.__new__(nn.Linear)
  source.weight,source.bias=Tensor.empty(columns,depth,dtype=dtype,device=device),None
  return source


def fixture(blocks=2):
  embedding=nn.Embedding.__new__(nn.Embedding)
  embedding.weight=Tensor.empty(16,2048,dtype=dtypes.half,device='CL')
  layers=[]
  for _ in range(blocks):
    block=GatedDeltaNetBlock.__new__(GatedDeltaNetBlock)
    block.head_k_dim=block.head_v_dim=128
    block.ffn_gate,block.ffn_up=linear(6144,2048),linear(6144,2048)
    block.ffn_down,block.attn_qkv=linear(2048,6144),linear(6144,2048)
    block.attn_gate,block.ssm_out=linear(2048,2048),linear(2048,2048)
    layers.append(block)
  return SimpleNamespace(token_embd=embedding,output=linear(16,2048),blk=layers,
                         config=SimpleNamespace(max_context=16),max_context=16)


class TestMaliGdnGateUp(unittest.TestCase):
  def setUp(self):
    device=patch.object(mali,'Device',{'CL':SimpleNamespace(device_name='Mali-LODX r0p0',driver_version='2.1',device_exts=())})
    device.start()
    self.addCleanup(device.stop)
    scan=patch('tinygrad.llm.kernels.mali_recurrent.supported',return_value=True)
    scan.start()
    self.addCleanup(scan.stop)
    packing=patch('tinygrad.llm.kernels.mali_linear.pack_weight',side_effect=Packed)
    self.pack=packing.start()
    self.addCleanup(packing.stop)

  def configure(self,model,**kwargs):
    return mali.configure(model,pack_weights=False,recurrent_scan=True,**kwargs)

  def test_default_off_enable_exact36_original_bindings_reuse_then_disable(self):
    model=fixture(18)
    plain=self.configure(model)
    self.assertEqual((plain['gdn_gate_up_packed_layers'],plain['gdn_gate_up_packed_bytes']),(0,0))
    self.pack.assert_not_called()
    original={(id(block),name):(getattr(block,name),getattr(block,name).weight,getattr(block,name).weight.uop)
              for block in model.blk for name in ('ffn_gate','ffn_up','ffn_down','attn_qkv','attn_gate','ssm_out')}
    enabled=self.configure(model,pack_gdn_gate_up=True)
    self.assertEqual((enabled['gdn_gate_up_packed_layers'],enabled['gdn_gate_up_packed_bytes']),(36,905969664))
    self.assertFalse(enabled['recurrent_pack_weights'])
    self.assertEqual(enabled['gdn_gate_up_packed_scope'],'ffn_gate,ffn_up')
    self.assertEqual(self.pack.call_count,36)
    packed=[]
    for block in model.blk:
      for name in ('ffn_gate','ffn_up','ffn_down','attn_qkv','attn_gate','ssm_out'):
        layer=getattr(block,name)
        old,weight,node=original[id(block),name]
        self.assertIs(layer,old)
        self.assertIs(layer.weight,weight)
        self.assertIs(layer.weight.uop,node)
        self.assertIs(layer.source.weight,weight)
        if name in ('ffn_gate','ffn_up'):
          self.assertEqual(layer.packed_weight.nbytes(),25165824)
          packed.append(layer.packed_weight)
        else:self.assertIsNone(layer.packed_weight)
    self.configure(model,pack_gdn_gate_up=True)
    self.assertEqual(self.pack.call_count,36)
    self.assertEqual([getattr(block,name).packed_weight for block in model.blk for name in ('ffn_gate','ffn_up')],packed)
    disabled=self.configure(model)
    self.assertEqual((disabled['gdn_gate_up_packed_layers'],disabled['gdn_gate_up_packed_bytes']),(0,0))
    self.assertTrue(all(getattr(block,name).packed_weight is None for block in model.blk for name in ('ffn_gate','ffn_up')))

  def test_disable_even_when_recurrent_scan_is_disabled_and_preserve_output_pack(self):
    model=fixture()
    self.configure(model,pack_output=True,pack_gdn_gate_up=True)
    head,head_weight,packed=model.output,model.output.weight,model.output.packed_weight
    disabled=mali.configure(model,pack_weights=False,recurrent_scan=False,pack_output=True)
    self.assertEqual(disabled['gdn_gate_up_packed_scope'],'off')
    self.assertTrue(all(getattr(block,name).packed_weight is None for block in model.blk for name in ('ffn_gate','ffn_up')))
    self.assertIs(model.output,head)
    self.assertIs(head.weight,head_weight)
    self.assertIs(head.packed_weight,packed)

  def test_stale_copy_is_released_then_rebuilt_without_weight_replacement(self):
    model=fixture(1)
    self.configure(model,pack_gdn_gate_up=True)
    layer=model.blk[0].ffn_gate
    original_weight,old_pack=layer.weight,layer.packed_weight
    layer.weight.replace(Tensor.empty(*layer.weight.shape,device='CL',dtype=dtypes.half))
    def rebuild(weight):
      self.assertIsNone(layer.packed_weight)
      self.assertIs(weight,original_weight)
      return Packed(weight)
    self.pack.side_effect=rebuild
    self.configure(model,pack_gdn_gate_up=True)
    self.assertIsNot(layer.packed_weight,old_pack)
    self.assertIs(layer.packed_source,original_weight.uop)
    self.assertEqual(self.pack.call_count,3)

  def test_all_candidate_guards_run_before_selected_allocations(self):
    for bad in ('dtype','shape','device','missing','bias','q8','no_scan','no_blocks'):
      model=fixture()
      self.configure(model)
      extra={}
      if bad=='dtype':model.blk[-1].ffn_up.weight=Tensor.empty(6144,2048,device='CL',dtype=dtypes.float)
      elif bad=='shape':model.blk[-1].ffn_up.weight=Tensor.empty(6143,2048,device='CL',dtype=dtypes.half)
      elif bad=='device':model.blk[-1].ffn_up.weight=Tensor.empty(6144,2048,device='CPU',dtype=dtypes.half)
      elif bad=='missing':del model.blk[-1].ffn_up
      elif bad=='bias':model.blk[-1].ffn_up.bias=Tensor.empty(6144,device='CL',dtype=dtypes.half)
      elif bad=='q8':extra['q8_projections_checkpoint']='never-open.gguf'
      elif bad=='no_scan':extra['recurrent_scan']=False
      else:model.blk=[]
      with self.subTest(bad=bad):
        with self.assertRaises(ValueError):
          mali.configure(model,pack_weights=False,pack_gdn_gate_up=True,**({'recurrent_scan':True}|extra))
        self.pack.assert_not_called()
    with self.assertRaises(TypeError):self.configure(fixture(),pack_gdn_gate_up='off')

  def test_allocation_failure_releases_new_copies_and_preserves_valid_borrowed_copy(self):
    model=fixture(1)
    self.configure(model)
    def fail_second(weight):
      if self.pack.call_count==2:raise MemoryError('synthetic pack failure')
      return Packed(weight)
    self.pack.side_effect=fail_second
    with self.assertRaises(MemoryError):self.configure(model,pack_gdn_gate_up=True)
    self.assertIsNone(model.blk[0].ffn_gate.packed_weight)
    self.assertIsNone(model.blk[0].ffn_up.packed_weight)
    self.pack.side_effect=Packed
    self.configure(model,pack_gdn_gate_up=True)
    gate=model.blk[0].ffn_gate.packed_weight
    model.blk[0].ffn_up.packed_weight=None
    self.pack.side_effect=MemoryError('synthetic replacement failure')
    with self.assertRaises(MemoryError):self.configure(model,pack_gdn_gate_up=True)
    self.assertIs(model.blk[0].ffn_gate.packed_weight,gate)
    self.assertIsNone(model.blk[0].ffn_up.packed_weight)

  def test_combined_head_pack_failure_releases_new_body_copies(self):
    model=fixture(1)
    self.configure(model)
    def pack(weight):
      if weight is model.output.weight:raise MemoryError('synthetic head pack failure')
      return Packed(weight)
    self.pack.side_effect=pack
    with self.assertRaises(MemoryError):self.configure(model,pack_output=True,pack_gdn_gate_up=True)
    self.assertIsNone(model.blk[0].ffn_gate.packed_weight)
    self.assertIsNone(model.blk[0].ffn_up.packed_weight)
    self.assertIsNone(model.output.packed_weight)

  def test_combined_head_pack_failure_preserves_preexisting_valid_body(self):
    model=fixture(1)
    self.configure(model,pack_gdn_gate_up=True)
    previous=(model.blk[0].ffn_gate.packed_weight,model.blk[0].ffn_up.packed_weight)
    self.pack.side_effect=MemoryError('synthetic head pack failure')
    with self.assertRaises(MemoryError):self.configure(model,pack_output=True,pack_gdn_gate_up=True)
    self.assertIs(model.blk[0].ffn_gate.packed_weight,previous[0])
    self.assertIs(model.blk[0].ffn_up.packed_weight,previous[1])
    self.assertIsNone(model.output.packed_weight)

  def test_late_attachment_failure_clears_new_packs_but_preserves_valid_head(self):
    class Wrapper:
      def __init__(self,source):self.source=source
      @property
      def weight(self):return self.source.weight
    class Raw:
      def to(self,device):return self
      def contiguous(self):return self
      def realize(self):return self
    for existing_head in (False,True):
      model=fixture(1)
      model.output=Wrapper(model.output)
      self.configure(model,pack_output=existing_head)
      head=model.output.source
      previous=head.packed_weight
      entries={'output.weight':(Raw(),(16,2048),8)}
      with self.subTest(existing_head=existing_head),patch('tinygrad.llm.gguf.gguf_parse',return_value=({},entries)):
        with self.assertRaisesRegex(ValueError,'Q8 head could not attach'):
          self.configure(model,pack_output=True,pack_gdn_gate_up=True,q8_head_checkpoint='never-open.gguf')
      self.assertIsNone(model.blk[0].ffn_gate.packed_weight)
      self.assertIsNone(model.blk[0].ffn_up.packed_weight)
      self.assertIs(head.packed_weight,previous)


if __name__=='__main__':unittest.main()
