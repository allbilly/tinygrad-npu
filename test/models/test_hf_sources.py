import hashlib, io, json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from extra.rockchip.fetch_benchmark_checkpoints import fetch_file
from extra.rockchip.fetch_hf_sources import git_blob, small_file, weight_files
from test.models.test_benchmark_download import Response


class TestHFSources(unittest.TestCase):
  def test_shards_keep_independent_identity_and_verified_payloads(self):
    payloads = {'part1.bin':b'first weight shard','part2.bin':b'second different shard'}
    source = {'model':'fixture','size':'small','source_model':'owner/model','repo':'owner/model','revision':'pinned'}
    calls = []
    def open_range(request,timeout):
      name = request.full_url.split('?')[0].rsplit('/',1)[1]
      calls.append(name)
      start,end = map(int,request.headers['Range'].removeprefix('bytes=').split('-'))
      data = payloads[name]
      return Response(data[start:end+1],start,end,len(data))
    with tempfile.TemporaryDirectory() as directory,patch('urllib.request.urlopen',side_effect=open_range):
      root,results = Path(directory),[]
      for name in ['part1.bin','part2.bin','part1.bin']:
        data = payloads[name]
        item = {'rfilename':name,'size':len(data),'lfs':{'sha256':hashlib.sha256(data).hexdigest()}}
        results.append(fetch_file(item,root,source,connections=1,metadata_name=name+'.download.json'))
      self.assertEqual(calls,['part1.bin','part2.bin'])
      for result in results:self.assertTrue(result['complete'])
      for name,data in payloads.items():
        self.assertEqual((root/name).read_bytes(),data)
        identity = json.loads((root/(name+'.download.json')).read_text())
        self.assertEqual(identity['filename'],name)
        self.assertEqual(identity['sha256'],hashlib.sha256(data).hexdigest())
      changed = {**source,'revision':'different'}
      with self.assertRaises(ValueError):fetch_file(item,root,changed,metadata_name='part1.bin.download.json')

  def test_small_snapshot_file_checks_git_blob_and_reuses_verified_copy(self):
    data = b'{"hidden_size":32}'
    item = {'rfilename':'config.json','size':len(data),'blobId':git_blob(data)}
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      with patch('urllib.request.urlopen',return_value=io.BytesIO(data)) as network:
        result = small_file('owner/model','pinned',item,root)
        self.assertEqual(network.call_count,1)
      with patch('urllib.request.urlopen',side_effect=AssertionError('Should reuse verified metadata')):
        self.assertEqual(small_file('owner/model','pinned',item,root),result)
      self.assertEqual(result['sha256'],hashlib.sha256(data).hexdigest())
      with self.assertRaises(ValueError):small_file('owner/model','changed',{**item,'blobId':'0'*40},root)
      self.assertEqual((root/'config.json').read_bytes(),data)

  def test_corrupt_small_file_never_becomes_snapshot_metadata(self):
    good,bad = b'good data',b'bad! data'
    item = {'rfilename':'config.json','size':len(good),'blobId':git_blob(good)}
    with tempfile.TemporaryDirectory() as directory,patch('urllib.request.urlopen',return_value=io.BytesIO(bad)):
      root = Path(directory)
      with self.assertRaises(ValueError):small_file('owner/model','pinned',item,root)
      self.assertFalse((root/'config.json').exists())

  def test_small_lfs_tokenizer_checks_payload_sha_not_pointer_blob(self):
    data = b'binary tokenizer payload'
    item = {'rfilename':'tokenizer.model','size':len(data),'blobId':'f'*40,
            'lfs':{'sha256':hashlib.sha256(data).hexdigest()}}
    with tempfile.TemporaryDirectory() as directory,patch('urllib.request.urlopen',return_value=io.BytesIO(data)):
      root = Path(directory)
      result = small_file('owner/model','pinned',item,root)
      self.assertEqual(result['verification'],'lfs_sha256')
      self.assertEqual((root/'tokenizer.model').read_bytes(),data)
      with self.assertRaises(ValueError):small_file('owner/model','pinned',{**item,'lfs':{'sha256':'0'*64}},root)

  def test_index_selects_only_referenced_shards_and_rejects_missing_or_nested_files(self):
    index = 'pytorch_model.bin.index.json'
    items = [{'rfilename':index},*({'rfilename':name,'lfs':{'sha256':'expected'}} for name in ('part1.bin','part2.bin','optimizer.bin'))]
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      (root/index).write_text(json.dumps({'weight_map':{'layer1':'part1.bin','layer2':'part2.bin'}}))
      self.assertEqual([item['rfilename'] for item in weight_files({'siblings':items},root)],['part1.bin','part2.bin'])
      with self.assertRaises(ValueError):weight_files({'siblings':items[:2]},root)
      (root/index).write_text(json.dumps({'weight_map':{'layer1':'../part1.bin'}}))
      with self.assertRaises(ValueError):weight_files({'siblings':items+[{'rfilename':'../part1.bin','lfs':{}}]},root)
