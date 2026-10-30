from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
from tinygrad import Device, Tensor, TinyJit, UOp, dtypes, nn
from tinygrad.helpers import round_up
from tinygrad.llm.model import ExpertWeights
from tinygrad.llm.kernels import rockchip_routed_int8
from tinygrad.llm.kernels.rockchip_npu_experts import NPUExperts, configure, route_map
from tinygrad.llm.kernels.rockchip_experts import CPUExpertWeights, configure as configure_cpu


def stable_reference(selected,experts,capacity,topk,broadcast):
  flat=selected.reshape(-1)
  counts=np.bincount(flat,minlength=experts).astype(np.int32)
  index=np.zeros(capacity*topk,dtype=np.int32)
  inverse=np.zeros((capacity,topk),dtype=np.int32)
  cursor=np.cumsum(np.concatenate(([0],counts[:-1]))).astype(np.int32)
  for i,expert in enumerate(flat):
    slot=cursor[expert]
    index[slot]=i//topk if broadcast else i
    inverse.flat[i]=slot
    cursor[expert]+=1
  return index,inverse,counts


def small_experts(bias=False):
  gate=ExpertWeights(4,64,32,bias=bias)
  up=ExpertWeights(4,64,32,bias=bias)
  down=ExpertWeights(4,32,64,bias=bias)
  for layer in (gate,up,down):
    layer.weight=Tensor.ones(*layer.weight.shape,device='CPU',dtype=dtypes.half).realize()
    if bias: layer.bias=Tensor.zeros(*layer.bias.shape,device='CPU',dtype=dtypes.half).realize()
  return gate,up,down


class TestNPURouteMap(unittest.TestCase):
  def test_symbolic_rows_fixed_capacity_broadcast_and_per_route(self):
    opened=set(Device._opened_devices)
    rng=np.random.default_rng(20261101)
    rowvar=UOp.variable('npu_expert_route_rows',1,17)
    cases=((1,np.array([[0,0]])),(7,np.array([[2,1],[2,2],[0,3],[1,0],[2,0],[1,3],[3,3]])),
           (17,rng.integers(0,4,(17,2),dtype=np.int32)),(3,np.array([[1,1],[0,2],[3,0]])))
    for broadcast in (True,False):
      run=TinyJit(lambda selected,rows:tuple(v.realize() for v in route_map(
        selected[:,:rows],4,broadcast=broadcast)))
      for rows,ids in cases:
        selected=Tensor(np.pad(ids.reshape(1,rows,2),((0,0),(0,17-rows),(0,0))),device='CPU').cast(dtypes.int).contiguous().realize()
        actual=run(selected,rowvar.bind(rows))
        expected=stable_reference(ids,4,17,2,broadcast)
        np.testing.assert_array_equal(actual[0].numpy(),expected[0])
        np.testing.assert_array_equal(actual[1].numpy(),expected[1])
        np.testing.assert_array_equal(actual[2].numpy(),expected[2])
        self.assertTrue((actual[0].numpy()<(17 if broadcast else 34)).all())
        self.assertTrue((actual[1].numpy()<34).all())
    self.assertFalse(any(device.split(':')[0]=='ROCKCHIP' for device in Device._opened_devices-opened))

  def test_large_expert_count_balanced_and_skewed_capacity(self):
    opened=set(Device._opened_devices)
    for ids in (np.arange(128*6,dtype=np.int32).reshape(1,128,6)%64,
                np.repeat(np.arange(6,dtype=np.int32),128).reshape(1,128,6)):
      selected=Tensor(ids,device='CPU').cast(dtypes.int).contiguous().realize()
      index,inverse,counts=route_map(selected,64,broadcast=False)
      expected=stable_reference(ids,64,128,6,False)
      np.testing.assert_array_equal(index.numpy(),expected[0])
      np.testing.assert_array_equal(inverse.numpy(),expected[1])
      np.testing.assert_array_equal(counts.numpy(),expected[2])
    self.assertFalse(any(device.split(':')[0]=='ROCKCHIP' for device in Device._opened_devices-opened))

  def test_invalid_ids_and_occupancy_overflow_are_bounded(self):
    negative=(np.arange(34,dtype=np.int32)%4).reshape(1,17,2)
    negative[0,-1,1]=-1
    last_expert=(np.arange(34,dtype=np.int32)%4).reshape(1,17,2)
    last_expert[0,-1,1]=4
    for ids in (negative,last_expert,np.zeros((1,17,2),dtype=np.int32)):
      selected=Tensor(ids,device='CPU').cast(dtypes.int).contiguous().realize()
      index,inverse,counts=route_map(selected,4,broadcast=True)
      self.assertGreater(int(counts.numpy()[0]),17)
      self.assertTrue(((index.numpy()>=0)&(index.numpy()<17)).all())
      self.assertTrue(((inverse.numpy()>=0)&(inverse.numpy()<17)).all())
      self.assertTrue((index.numpy()==0).all())
      self.assertTrue((inverse.numpy()==0).all())


class TestNPUExpertConfigure(unittest.TestCase):
  @staticmethod
  def model(bias=False):
    gate,up,down=small_experts(bias)
    block=SimpleNamespace(ffn_gate_exps=gate,ffn_up_exps=up,ffn_down_exps=down,
                          config=SimpleNamespace(num_experts_per_tok=2))
    return SimpleNamespace(blk=[block]),(gate,up,down)

  def test_configure_preserves_state_skips_cpu_copy_then_cpu_can_wrap(self):
    opened=set(Device._opened_devices)
    model,layers=self.model()
    before=nn.state.get_state_dict(model)
    identities={name:id(value) for name,value in before.items()}
    state_names=set(before)
    calls=[]
    def fake_pack(weight):
      calls.append(weight)
      experts,columns,depth=weight.shape
      packed=Tensor.zeros(experts,round_up(columns,32)//32,depth//32,32,32,device='CPU',dtype=dtypes.int8).realize()
      scales=Tensor.ones(experts,round_up(columns,32),device='CPU',dtype=dtypes.float).realize()
      return packed,scales
    with patch.object(rockchip_routed_int8,'pack_weight',side_effect=fake_pack):
      count,packed_bytes=configure(model,enabled=True)
    self.assertEqual(count,1)
    self.assertGreater(packed_bytes,0)
    self.assertEqual(len(calls),2)
    block=model.blk[0]
    self.assertIsInstance(block.npu_experts,NPUExperts)
    self.assertEqual({name:id(value) for name,value in nn.state.get_state_dict(model).items()},identities)
    self.assertEqual(set(nn.state.get_state_dict(model)),state_names)
    self.assertIs(block.npu_experts._sources()[0],layers[0])
    self.assertIs(block.npu_experts._sources()[1],layers[1])
    self.assertIs(block.npu_experts._sources()[2],layers[2])

    self.assertEqual(configure_cpu(model,enabled=True),0)
    for name,layer in zip(('ffn_gate_exps','ffn_up_exps','ffn_down_exps'),layers):
      self.assertIs(getattr(block,name),layer)
    self.assertEqual(configure(model,enabled=False),(0,0))
    self.assertFalse(hasattr(block,'npu_experts'))
    self.assertEqual(configure_cpu(model,enabled=True),3)
    for name,layer in zip(('ffn_gate_exps','ffn_up_exps','ffn_down_exps'),layers):
      wrapper=getattr(block,name)
      self.assertIsInstance(wrapper,CPUExpertWeights)
      self.assertIs(wrapper._original(),layer)
    self.assertEqual({name:id(value) for name,value in nn.state.get_state_dict(model).items()},identities)
    self.assertEqual(set(nn.state.get_state_dict(model)),state_names)
    self.assertFalse(any(device.split(':')[0]=='ROCKCHIP' for device in Device._opened_devices-opened))

  def test_bad_bool_bias_and_geometry_rejected_before_pack(self):
    model,_=self.model()
    packed=[]
    with patch.object(rockchip_routed_int8,'pack_weight',side_effect=lambda w:packed.append(w)):
      with self.assertRaises(ValueError): configure(model,enabled=1)
    self.assertEqual(packed,[])
    model,layers=self.model(bias=True)
    with patch.object(rockchip_routed_int8,'pack_weight',side_effect=lambda w:packed.append(w)):
      self.assertEqual(configure(model,enabled=True),(0,0))
    self.assertEqual(packed,[])
    model,layers=self.model()
    layers[1].weight=layers[1].weight.float()
    with patch.object(rockchip_routed_int8,'pack_weight',side_effect=lambda w:packed.append(w)):
      self.assertEqual(configure(model,enabled=True),(0,0))
    self.assertEqual(packed,[])

  def test_weight_residual_requires_activation_correction_and_preserves_state(self):
    opened=set(Device._opened_devices)
    model,_=self.model()
    for kwargs in ({'weight_residual':True},{'activation_residual':True,'weight_residual':1}):
      with self.assertRaises(ValueError):configure(model,enabled=True,**kwargs)
    identities={name:id(value) for name,value in nn.state.get_state_dict(model).items()}
    calls=[]
    def fake_pair(weight):
      calls.append(weight.shape)
      experts,columns,depth=weight.shape
      packed=Tensor.zeros(experts,round_up(columns,32)//32,depth//32,32,32,device='CPU',dtype=dtypes.int8).realize()
      scale=Tensor.ones(experts,round_up(columns,32),device='CPU',dtype=dtypes.float).realize()
      return packed,scale,packed.clone().realize(),scale.clone().realize()
    with patch.object(rockchip_routed_int8,'pack_weight_residual',side_effect=fake_pair):
      count,packed_bytes=configure(model,enabled=True,activation_residual=True,weight_residual=True)
    helper=model.blk[0].npu_experts
    self.assertEqual(count,1)
    self.assertEqual(len(calls),2)
    base=helper._packed()
    extra=helper._residual_packed()
    self.assertEqual(packed_bytes,base[0].nbytes()+base[2].nbytes()+extra[0].nbytes()+extra[2].nbytes())
    self.assertIsNot(base[0],extra[0])
    self.assertEqual({name:id(value) for name,value in nn.state.get_state_dict(model).items()},identities)
    self.assertFalse(any(device.split(':')[0]=='ROCKCHIP' for device in Device._opened_devices-opened))

  def test_enabled_shape_guard(self):
    layers=small_experts()
    helper=NPUExperts(*layers,topk=2)
    selected=Tensor.zeros(1,7,2,device='CPU',dtype=dtypes.int).realize()
    x=Tensor.ones(1,7,1,64,device='CPU',dtype=dtypes.float).realize()
    self.assertTrue(helper.enabled(selected,x))
    self.assertFalse(helper.enabled(selected,x.half()))
    self.assertFalse(helper.enabled(selected,x.expand(1,7,2,64).contiguous()))
    self.assertFalse(helper.enabled(selected[:,:,:1],x))

  def test_cpu_down_exception_preserves_npu_gate_up_and_state(self):
    opened=set(Device._opened_devices)
    model,layers=self.model()
    identities={name:id(value) for name,value in nn.state.get_state_dict(model).items()}
    state_names=set(identities)
    calls=[]
    def fake_pack(weight):
      calls.append(weight)
      experts,columns,depth=weight.shape
      packed=Tensor.zeros(experts,round_up(columns,32)//32,depth//32,32,32,device='CPU',dtype=dtypes.int8).realize()
      scales=Tensor.ones(experts,round_up(columns,32),device='CPU',dtype=dtypes.float).realize()
      return packed,scales
    with patch.object(rockchip_routed_int8,'pack_weight',side_effect=fake_pack):
      self.assertEqual(configure(model,enabled=True,cpu_down_layers=(0,))[0],1)
      block=model.blk[0]
      helper=block.npu_experts
      self.assertEqual(len(calls),1)
      self.assertTrue(helper.owns('ffn_gate_exps'))
      self.assertTrue(helper.owns('ffn_up_exps'))
      self.assertFalse(helper.owns('ffn_down_exps'))
      self.assertIsNone(helper._packed()[2])
      self.assertIsNone(helper._packed()[3])
      self.assertEqual(configure_cpu(model,enabled=True),1)
    self.assertIs(block.ffn_gate_exps,layers[0])
    self.assertIs(block.ffn_up_exps,layers[1])
    self.assertIsInstance(block.ffn_down_exps,CPUExpertWeights)
    self.assertIs(block.ffn_down_exps._original(),layers[2])
    self.assertIs(helper._cpu_down(),block.ffn_down_exps)
    self.assertEqual({name:id(value) for name,value in nn.state.get_state_dict(model).items()},identities)
    self.assertEqual(set(nn.state.get_state_dict(model)),state_names)
    self.assertFalse(any(device.split(':')[0]=='ROCKCHIP' for device in Device._opened_devices-opened))

  def test_full_cpu_block_and_bad_indices_before_pack_even_when_disabled(self):
    model,layers=self.model()
    calls=[]
    def fake_pack(weight):
      calls.append(weight)
      experts,columns,depth=weight.shape
      packed=Tensor.zeros(experts,round_up(columns,32)//32,depth//32,32,32,device='CPU',dtype=dtypes.int8).realize()
      scales=Tensor.ones(experts,round_up(columns,32),device='CPU',dtype=dtypes.float).realize()
      return packed,scales
    with patch.object(rockchip_routed_int8,'pack_weight',side_effect=fake_pack):
      self.assertEqual(configure(model,enabled=True,cpu_layers=(0,)),(0,0))
      self.assertEqual(calls,[])
      self.assertEqual(configure_cpu(model,enabled=True),3)
      self.assertEqual(configure(model,enabled=False,cpu_layers=(0,)),(0,0))
    self.assertFalse(hasattr(model.blk[0],'npu_experts'))
    for name,layer in zip(('ffn_gate_exps','ffn_up_exps','ffn_down_exps'),layers):
      self.assertIsInstance(getattr(model.blk[0],name),CPUExpertWeights)
      self.assertIs(getattr(model.blk[0],name)._original(),layer)

    model,layers=self.model()
    with patch.object(rockchip_routed_int8,'pack_weight',side_effect=fake_pack):
      self.assertEqual(configure(model,enabled=True,cpu_down_layers=(0,))[0],1)
      helper=model.blk[0].npu_experts
      for key,invalid in (('cpu_layers',[]),('cpu_down_layers',[-1]),('cpu_layers',(1,)),
                          ('cpu_down_layers',(True,)),('cpu_layers',('0',))):
        with self.subTest(key=key,invalid=invalid),self.assertRaises(ValueError):
          configure(model,enabled=True,**{key:invalid})
        self.assertEqual(len(calls),1)
        with self.assertRaises(ValueError): configure(model,enabled=False,**{key:invalid})
        self.assertIs(model.blk[0].npu_experts,helper)
      self.assertEqual(configure(model,enabled=False,cpu_layers=(0,),cpu_down_layers=(0,)),(0,0))
    self.assertFalse(hasattr(model.blk[0],'npu_experts'))


if __name__=='__main__': unittest.main()
