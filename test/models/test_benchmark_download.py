import hashlib, io, json, tempfile, threading, unittest, urllib.error
from pathlib import Path
from unittest.mock import patch
from extra.rockchip.fetch_benchmark_checkpoints import fetch, fetch_selected


class Response(io.BytesIO):
  def __init__(self, payload, start, end, size, status=206):
    super().__init__(payload)
    self.status,self.headers = status,{'Content-Range':f'bytes {start}-{end}/{size}'}


class TestBenchmarkDownload(unittest.TestCase):
  def row(self, payload):
    return {'model':'fixture','size':'small','checkpoint_candidate':'fixture/source',
            'gguf_probe':{'resolved_repo':'fixture/quant','revision':'pinned',
                          'files':[{'rfilename':'model.gguf','size':len(payload),'lfs':{'sha256':hashlib.sha256(payload).hexdigest()}}]}}

  def test_truncated_connection_resumes_the_actual_prefix_and_checks_hash(self):
    payload = b'checkpoint data whose connection closes early'
    offsets = []
    def open_range(request,timeout):
      start,end = map(int,request.headers['Range'].removeprefix('bytes=').split('-'))
      offsets.append(start)
      return Response(payload[start:10] if len(offsets)==1 else payload[start:end+1],start,end,len(payload))
    with tempfile.TemporaryDirectory() as directory,patch('urllib.request.urlopen',side_effect=open_range),patch('time.sleep'):
      result = fetch(self.row(payload),Path(directory),connections=1)
      self.assertTrue(result['complete'])
      self.assertEqual(offsets,[0,10])
      self.assertEqual(Path(result['checkpoint']).read_bytes(),payload)
      self.assertEqual(result['sha256'],hashlib.sha256(payload).hexdigest())

  def test_ignored_range_or_corrupt_payload_never_becomes_a_checkpoint(self):
    payload = b'expected checkpoint'
    for status,actual in ((200,payload),(206,b'x'*len(payload))):
      with self.subTest(status=status),tempfile.TemporaryDirectory() as directory:
        with patch('urllib.request.urlopen',return_value=Response(actual,0,len(payload)-1,len(payload),status)):
          result = fetch(self.row(payload),Path(directory),connections=1)
        self.assertFalse(result['complete'])
        self.assertFalse(Path(result['checkpoint']).exists())

  def test_parallel_segments_commit_in_order_and_resume_partial_prefix(self):
    payload = bytes(range(32))
    finished = []
    later_finished = threading.Event()
    def open_range(request,timeout):
      start,end = map(int,request.headers['Range'].removeprefix('bytes=').split('-'))
      if start in (0,3):self.assertTrue(later_finished.wait(10),'Later segment did not finish')
      finished.append(start)
      if start not in (0,3):later_finished.set()
      return Response(payload[start:end+1],start,end,len(payload))
    with tempfile.TemporaryDirectory() as directory,patch('urllib.request.urlopen',side_effect=open_range):
      result = fetch(self.row(payload),Path(directory),connections=4,range_bytes=8)
      self.assertTrue(result['complete'])
      self.assertNotEqual(finished[0],0)
      target = Path(result['checkpoint'])
      self.assertEqual(target.read_bytes(),payload)
      target.rename(target.with_suffix('.partial'))
      target.with_suffix('.partial').write_bytes(payload[:3])
      finished.clear()
      later_finished.clear()
      resumed = fetch(self.row(payload),Path(directory),connections=4,range_bytes=8)
      self.assertTrue(resumed['complete'])
      self.assertEqual(sorted(finished),[3,11,19,27])
      self.assertEqual(target.read_bytes(),payload)
      self.assertFalse((target.parent/(target.name+'.parts')).exists())

  def test_repeated_short_connections_keep_verified_byte_progress(self):
    payload = b'continuously making progress'
    offsets = []
    def open_range(request,timeout):
      start,end = map(int,request.headers['Range'].removeprefix('bytes=').split('-'))
      offsets.append(start)
      return Response(payload[start:min(start+4,end+1)],start,end,len(payload))
    with tempfile.TemporaryDirectory() as directory,patch('urllib.request.urlopen',side_effect=open_range),patch('time.sleep'):
      result = fetch(self.row(payload),Path(directory),connections=1)
      self.assertTrue(result['complete'])
      self.assertGreater(len(offsets),3)
      self.assertEqual(Path(result['checkpoint']).read_bytes(),payload)

  def test_retry_pass_skips_completed_models_and_records_verified_completion(self):
    payload = b'verified model data'
    stable,flaky = self.row(payload),self.row(payload)
    flaky['model'] = 'flaky'
    flaky['gguf_probe']['files'][0]['rfilename'] = 'flaky.gguf'
    calls = {'stable':0,'flaky':0}
    def open_range(request,timeout):
      key = 'flaky' if '/flaky.gguf?' in request.full_url else 'stable'
      calls[key]+=1
      if key=='flaky' and calls[key]<=3:raise urllib.error.URLError('transient outage')
      start,end = map(int,request.headers['Range'].removeprefix('bytes=').split('-'))
      return Response(payload[start:end+1],start,end,len(payload))
    with tempfile.TemporaryDirectory() as directory,patch('urllib.request.urlopen',side_effect=open_range),patch('time.sleep'):
      root = Path(directory)
      report = fetch_selected([stable,flaky],root,workers=1,connections=1,rounds=3,retry_delay=0)
      self.assertTrue(report['complete'])
      self.assertEqual(report['round'],2)
      self.assertEqual(calls,{'stable':1,'flaky':4})
      self.assertEqual(json.loads(next(root.glob('downloads-*.json')).read_text()),report)
      for row in report['models']:
        self.assertEqual(Path(row['checkpoint']).read_bytes(),payload)
        self.assertEqual(row['sha256'],hashlib.sha256(payload).hexdigest())

  def test_retry_limit_keeps_incomplete_model_explicit(self):
    with tempfile.TemporaryDirectory() as directory,patch('urllib.request.urlopen',side_effect=urllib.error.URLError('offline')) as network:
      with patch('time.sleep'):
        report = fetch_selected([self.row(b'model')],Path(directory),workers=1,connections=1,rounds=2,retry_delay=0)
      self.assertFalse(report['complete'])
      self.assertEqual(report['round'],2)
      self.assertEqual(network.call_count,6)
      self.assertFalse(Path(report['models'][0]['checkpoint']).exists())
