"""Experimental fused INT8 experts with runtime compact routing maps."""
from __future__ import annotations
import functools
from dataclasses import replace
from typing import Callable
from tinygrad import Tensor,dtypes,UOp
from tinygrad.llm.kernels.rockchip_cpu import _renderer
from tinygrad.llm.kernels import rockchip_routed_int8 as routed,rockchip_bucketed_int8 as bucketed
from tinygrad.uop.ops import KernelInfo,Ops,ProgramInfo


@functools.cache
def _map_program(index:UOp,inverse:UOp,counts:UOp,selected:UOp,rows:UOp,experts:int,capacity:int,topk:int,broadcast:bool) -> UOp:
  routes=capacity*topk
  source=f'''void rockchip_expert_route_map(int *index,int *inverse,int *counts,const int *selected,const int *row_count) {{
    for(int i=0;i<{routes};i++){{index[i]=0;inverse[i]=0;}}
    for(int e=0;e<{experts};e++)counts[e]=0;
    int rows=*row_count;
    if(rows<1 || rows>{capacity}){{counts[0]={capacity+1};return;}}
    for(int i=0;i<rows*{topk};i++){{
      int e=selected[i];
      if((unsigned)e>={experts}U){{counts[0]={capacity+1};return;}}
      if(++counts[e]>{capacity}){{counts[0]={capacity+1};return;}}
    }}
    int cursor[{experts}],offset=0;
    for(int e=0;e<{experts};e++){{cursor[e]=offset;offset+=counts[e];}}
    for(int i=0;i<rows*{topk};i++){{int slot=cursor[selected[i]]++;
      index[slot]=i{"/"+str(topk) if broadcast else ""};inverse[i]=slot;}}
  }}'''
  sink=UOp.sink(index,inverse,counts,selected,rows,arg=KernelInfo('rockchip_expert_route_map'))
  renderer=_renderer()
  info=replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,1,2),ins=(3,4))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=(index,inverse,counts,selected,rows)),UOp(Ops.SOURCE,arg=source),
    UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(source))))


def route_map(selected:Tensor,experts:int,*,broadcast:bool) -> tuple[Tensor,Tensor,Tensor]:
  """Stable expert buckets. Invalid IDs/occupancy mark counts invalid for the NPU guard."""
  if (selected.device!='CPU' or selected.dtype is not dtypes.int or selected.ndim!=3 or selected.shape[0]!=1
      or type(experts) is not int or not 1<=experts<=128 or type(broadcast) is not bool
      or not isinstance(topk:=selected.shape[2],int) or not 1<=topk<=8 or not 1<=selected.max_shape[1]<=128):
    raise ValueError('Unsupported expert routing map geometry')
  rows,capacity=selected.shape[1],selected.max_shape[1]
  assert isinstance(capacity,int)
  indices=Tensor.empty(capacity*topk,device='CPU',dtype=dtypes.int)
  inverse=Tensor.empty(capacity,topk,device='CPU',dtype=dtypes.int)
  counts=Tensor.empty(experts,device='CPU',dtype=dtypes.int)
  number=Tensor(rows,device='CPU',dtype=dtypes.int).reshape(1).clone()
  result=indices.custom_kernel(inverse,counts,selected.pad_to((1,capacity,topk)).contiguous(),number,
    fxn=lambda i,v,c,s,n:_map_program(i,v,c,s,n,experts,capacity,topk,broadcast))
  return result[0],result[1],result[2]


class NPUExperts:
  """Original tensors remain on the block; runtime packed storage lives in closures."""
  def __init__(self,gate,up,down,topk:int,*,cpu_down:bool=False,activation_residual:bool=False,weight_residual:bool=False):
    self.experts,self.hidden,self.depth=gate.weight.shape
    assert isinstance(self.experts,int) and isinstance(self.hidden,int) and isinstance(self.depth,int)
    self.topk=topk
    self.activation_residual=activation_residual
    self.weight_residual=weight_residual
    self._sources=lambda:(gate,up,down)
    self._cpu_down=lambda:down
    self.use_npu_down=not cpu_down
    self._packed:Callable[[],tuple[Tensor,Tensor,Tensor|None,Tensor|None]|None]=lambda:None
    self._residual_packed:Callable[[],tuple[Tensor|None,Tensor|None,Tensor|None,Tensor|None]]=lambda:(None,None,None,None)
    self.packed_bytes=0

  def prepare(self):
    gate,up,down=self._sources()
    if self.weight_residual:
      gu,gs,gu1,gs1=routed.pack_weight_residual(gate.weight.cat(up.weight,dim=1))
      dw,ds,dw1,ds1=routed.pack_weight_residual(down.weight) if self.use_npu_down else (None,None,None,None)
    else:
      gu,gs=routed.pack_weight(gate.weight.cat(up.weight,dim=1))
      dw,ds=routed.pack_weight(down.weight) if self.use_npu_down else (None,None)
      gu1,gs1,dw1,ds1=None,None,None,None
    for tensor in (gu,dw,gs,ds,gu1,gs1,dw1,ds1):
      if tensor is not None:tensor.realize()
    self._packed=lambda:(gu,gs,dw,ds)
    self._residual_packed=lambda:(gu1,gs1,dw1,ds1)
    self.packed_bytes=sum(tensor.nbytes() for tensor in (gu,dw,gu1,dw1) if tensor is not None)

  def owns(self,name:str) -> bool:
    return name in ('ffn_gate_exps','ffn_up_exps') or name=='ffn_down_exps' and self.use_npu_down

  def enabled(self,selected:Tensor,x:Tensor) -> bool:
    return (x.device=='CPU' and x.dtype is dtypes.float and x.ndim==4 and x.shape[0]==1 and x.shape[2]==1
      and x.shape[-1]==self.depth and 1<=x.max_shape[1]<=128 and selected.device=='CPU' and selected.dtype is dtypes.int
      and selected.shape==(1,x.shape[1],self.topk))

  def _project(self,selected:Tensor,x:Tensor,weight:Tensor,scales:Tensor,columns:int,extra:tuple[Tensor,Tensor]|None=None) -> Tensor:
    rows,capacity=x.shape[1],x.max_shape[1]
    assert isinstance(capacity,int)
    broadcast=x.shape[2]==1
    if capacity==1:
      inputs=x.reshape(x.shape[2],x.shape[3]).expand(self.topk,x.shape[3]).contiguous()
      return routed.matmul(inputs,selected.reshape(self.topk),weight,scales,columns,
                           activation_residual=self.activation_residual,weight_residual=extra).reshape(1,1,self.topk,columns)
    index,inverse,counts=route_map(selected,self.experts,broadcast=broadcast)
    dense=x.pad_to((1,capacity,x.shape[2],x.shape[3])).reshape(capacity*x.shape[2],x.shape[3])
    result=bucketed.compact_matmul(dense[index],counts,weight,scales,columns,capacity,
                                  activation_residual=self.activation_residual,weight_residual=extra)[inverse]
    return result.reshape(1,capacity,self.topk,columns)[:,:rows]

  def gate_up(self,selected:Tensor,x:Tensor) -> tuple[Tensor,Tensor]:
    packed=self._packed()
    if packed is None:raise RuntimeError('Prepare NPU expert weights before capture')
    extra=self._residual_packed()
    result=self._project(selected,x,packed[0],packed[1],self.hidden*2,
                         (extra[0],extra[1]) if extra[0] is not None and extra[1] is not None else None)
    return result[...,:self.hidden],result[...,self.hidden:]

  def down(self,selected:Tensor,x:Tensor) -> Tensor:
    if not self.use_npu_down:return self._cpu_down()(selected,x)
    packed=self._packed()
    if packed is None:raise RuntimeError('Prepare NPU expert weights before capture')
    assert packed[2] is not None and packed[3] is not None
    extra=self._residual_packed()
    return self._project(selected,x,packed[2],packed[3],self.depth,
                         (extra[2],extra[3]) if extra[2] is not None and extra[3] is not None else None)


def configure(model,enabled:bool=False,*,cpu_layers:tuple[int,...]=(),cpu_down_layers:tuple[int,...]=(),
              activation_residual:bool=False,weight_residual:bool=False) -> tuple[int,int]:
  """Prepare supported CPU/Half expert blocks before capture; original state keys stay intact."""
  from tinygrad.llm.model import ExpertWeights
  from tinygrad.llm.kernels.rockchip_experts import CPUExpertWeights
  if type(enabled) is not bool:raise ValueError('NPU expert selection must be a bool')
  if type(activation_residual) is not bool:raise ValueError('Activation residual selection must be a bool')
  if type(weight_residual) is not bool or weight_residual and not activation_residual:
    raise ValueError('Weight residuals require activation residuals and a bool selection')
  blocks=getattr(model,'blk',[])
  for indices in (cpu_layers,cpu_down_layers):
    if type(indices) is not tuple or any(type(i) is not int or not 0<=i<len(blocks) for i in indices):
      raise ValueError('NPU expert CPU exceptions must be valid zero-based block indices')
  candidates=[]
  for block_index,block in enumerate(blocks):
    if isinstance(getattr(block,'npu_experts',None),NPUExperts):delattr(block,'npu_experts')
    if not enabled or block_index in cpu_layers:continue
    layers=[]
    for name in ('ffn_gate_exps','ffn_up_exps','ffn_down_exps'):
      layer=getattr(block,name,None)
      if isinstance(layer,CPUExpertWeights):
        layer=layer._original()
        setattr(block,name,layer)
      layers.append(layer)
    if not all(isinstance(layer,ExpertWeights) for layer in layers):continue
    gate,up,down=layers
    assert isinstance(gate,ExpertWeights) and isinstance(up,ExpertWeights) and isinstance(down,ExpertWeights)
    weights=[gate.weight,up.weight,down.weight]
    if any(w.device!='CPU' or w.dtype is not dtypes.half or w.ndim!=3
           or not all(isinstance(n,int) and n>0 for n in w.shape) for w in weights):continue
    e,h,d=gate.weight.shape
    assert isinstance(e,int) and isinstance(h,int) and isinstance(d,int)
    topk=getattr(getattr(block,'config',None),'num_experts_per_tok',0)
    if (not 1<=e<=128 or not 1<=topk<=8 or h*2>4096 or d>2048 or h<4 or d<4
        or up.weight.shape!=(e,h,d) or down.weight.shape!=(e,d,h) or any(hasattr(layer,'bias') for layer in layers)):continue
    helper=NPUExperts(gate,up,down,topk,cpu_down=block_index in cpu_down_layers,
                      activation_residual=activation_residual,weight_residual=weight_residual)
    helper._cpu_down=functools.partial(getattr,block,'ffn_down_exps')
    candidates.append((block,helper))
  for block,helper in candidates:
    helper.prepare()
    setattr(block,'npu_experts',helper)
  return len(candidates),sum(helper.packed_bytes for _,helper in candidates)
