"""Build dynamic INT8 expert chains from qualified fixed-row register templates."""
from __future__ import annotations
import ctypes,functools
from typing import Any
from tinygrad.device import Device,TinyELF
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.runtime.ops_rockchip import _cmd
from tinygrad.runtime.support.rockchip_int8 import int8_body


@functools.cache
def updater(groups:int,capacity:int,columns:int,depth:int,cores:int,rows:int|None=None,tile:int|None=None) -> Any:
  tile=capacity if tile is None else tile
  bodies=[int8_body(m,columns,depth,0,0,0,columns) for m in range(1,tile+1)]
  length=len(bodies[0])
  assert all(len(body)==length for body in bodies)
  offsets={pair:next(i for i,w in enumerate(bodies[0]) if (w>>48,w&0xffff)==pair)
           for pair in ((0x201,0x1070),(0x201,0x1110),(0x1001,0x4020))}
  stride=(length+5)//2*2
  table=',\n'.join('{'+','.join(f'{w}ULL' for w in body)+'}' for body in bodies)
  storage_rows=groups*capacity if rows is None else rows
  header=((storage_rows+groups)*4+63)//64*64
  source=f'''#include <stddef.h>
  typedef unsigned long long U64;
  static const U64 templates[{tile}][{length}]={{{table}}};
  int update_bucketed(void *commands,U64 command_dma,U64 lhs,U64 rhs,U64 out,const int *counts,unsigned *state) {{
    int total=0,used=0,offsets[{groups}];
    for(int g=0;g<{groups};g++){{
      if((unsigned)counts[g]>{capacity}U)return 1;
      offsets[g]={"used" if rows is not None else f"g*{capacity}"};
      used+=counts[g];if(used>{storage_rows})return 1;total+=(counts[g]+{tile-1})/{tile};
    }}
    if(!total)return 1;
    int active=({cores}==3 && total>=3)?3:1;
    for(int i=0;i<12;i++)state[i]=0;
    state[0]=total;state[1]=(1U<<active)-1;
    int begin=0;
    for(int core=0;core<active;core++){{
      int number=(total+active-1-core)/active;
      int range=active==1?2:6+core*2;
      state[range]=begin;state[range+1]=number;
      int ordinal=0,slot=begin;
      for(int g=0;g<{groups};g++){{
       for(int first=0;first<counts[g];first+={tile}){{
        if(ordinal++%active!=core)continue;
        int count=counts[g]-first;if(count>{tile})count={tile};
        U64 *words=(U64*)commands+(size_t)slot*{stride};
        __builtin_memcpy(words,templates[count-1],{length}*sizeof(U64));
        U64 a=lhs+{header}ULL+(U64)(offsets[g]+first)*{depth}ULL;
        U64 b=rhs+(U64)g*{columns*depth}ULL;
        U64 c=out+(U64)(offsets[g]+first)*{columns*4}ULL;
        words[{offsets[0x201,0x1070]}]={_cmd(0x201,0x1070)}ULL|(a<<16);
        words[{offsets[0x201,0x1110]}]={_cmd(0x201,0x1110)}ULL|(b<<16);
        words[{offsets[0x1001,0x4020]}]={_cmd(0x1001,0x4020)}ULL|(c<<16);
        if(slot+1<begin+number){{
          words[{length}]={_cmd(rk.TARGET_PC_REG,rk.REG_PC_BASE_ADDRESS)}ULL|
            (((command_dma+(U64)(slot+1)*{stride*8}ULL)&0xfffffff0ULL)<<16);
          words[{length+1}]={_cmd(rk.TARGET_PC_REG,rk.REG_PC_REGISTER_AMOUNTS,(length+1)//2+1)}ULL;
        }}else{{
          words[{length}]={_cmd(1,0)}ULL;
          words[{length+1}]={_cmd(rk.TARGET_PC_REG,rk.REG_PC_REGISTER_AMOUNTS)}ULL;
        }}
        words[{length+2}]={_cmd(rk.TARGET_VERSION,0)}ULL;
        words[{length+3}]={_cmd(rk.TARGET_PC,rk.REG_PC_OPERATION_ENABLE,0xd)}ULL;
        slot++;
       }}
      }}
      begin+=number;
    }}
    return 0;
  }}'''
  from tinygrad.llm.kernels.rockchip_cpu import _renderer
  renderer=_renderer()
  program:Any=Device['CPU'].runtime(TinyELF(renderer.compiler.compile_cached(source),'update_bucketed',renderer.target,()))
  program.fxn.restype=ctypes.c_int
  return program
