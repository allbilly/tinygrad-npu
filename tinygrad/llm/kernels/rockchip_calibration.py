"""Checkpoint-bound input-channel scales for existing INT8 projections."""
import array, gzip, json, math
from pathlib import Path
from tinygrad import Tensor, dtypes


def load_input_scales(path:str|Path, checkpoint_sha256:str) -> dict[str,Tensor]:
  path=Path(path)
  with (gzip.open(path,'rt',encoding='utf-8') if path.suffix=='.gz' else path.open(encoding='utf-8')) as stream:
    data=json.load(stream)
  if data.get('schema_version')!=1 or data.get('checkpoint_sha256')!=checkpoint_sha256:
    raise ValueError('Text input calibration does not match the checkpoint')
  result:dict[str,Tensor]={}
  for group in data['groups'].values():
    try:values=array.array('f',group['scales'])
    except (TypeError,ValueError,OverflowError) as exc:
      raise ValueError('Input channel scales must be finite positive vectors') from exc
    if not values or any(not math.isfinite(value) or value<=0 for value in values):
      raise ValueError('Input channel scales must be finite positive vectors')
    scale=Tensor(list(values),device='CPU',dtype=dtypes.float)
    for name in group['members']:
      if not isinstance(name,str) or not name:raise ValueError('Invalid calibrated projection name')
      if name in result:raise ValueError('Duplicate calibrated projection')
      result[name]=scale
  return result
