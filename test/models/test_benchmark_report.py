import copy, math, unittest
from extra.rockchip.benchmark_report import pair, render, render_prefill, summarize


class TestBenchmarkReport(unittest.TestCase):
  def fixture(self):
    return {'complete':True,'context':512,'input_ids':list(range(128)),'device':'CL','precision':'FP16 stored weights',
            'checkpoint_sha256':'weights','source_sha256':{'kernel':'source'},
            'trials':[{'warmup':i<3,'prompt_tokens':128,'generated_ids':list(range(64)),
                       'first_token_ms':200+i,'decode_tps':10+i,'peak_rss_bytes':(100+i)*1024**2} for i in range(6)]}

  def test_actual_rows_and_medians(self):
    result = summarize(self.fixture())
    self.assertEqual((result['ttft_ms'],result['decode_tps'],result['peak_rss_mib']),(204,14,105))
    self.assertFalse(result['qualified_win'])

  def test_reject_invalid_or_incomplete_measurements(self):
    for change in ('complete','warmups','output','input','nan','negative','nondeterministic','source'):
      report = copy.deepcopy(self.fixture())
      if change=='complete':report['complete']=False
      elif change=='warmups':report['trials']=report['trials'][1:]
      elif change=='output':report['trials'][-1]['generated_ids'].pop()
      elif change=='input':report['input_ids'].pop()
      elif change=='nan':report['trials'][-1]['decode_tps']=math.nan
      elif change=='negative':report['trials'][-1]['first_token_ms']=-1
      elif change=='nondeterministic':report['trials'][-1]['generated_ids'][0]=999
      else:report.pop('source_sha256')
      with self.subTest(change=change),self.assertRaises(ValueError):summarize(report)

  def test_pair_only_bolds_measured_numeric_leader(self):
    self.assertEqual(pair(10,None),'10.00 / —')
    self.assertEqual(pair(10,11),'10.00 / **11.00**')
    self.assertEqual(pair(10,11,lower=True),'**10.00** / 11.00')
    self.assertEqual(pair(10,10),'10.00 / 10.00')

  def test_row_identity_and_memory_metric(self):
    model = {'model':'Qwen3','size':'0.6B','dtype':'w8a8','published_ttft_ms':199.09,
             'published_decode_tps':32.91,'published_memory_mb':791.11}
    result = summarize(self.fixture())
    table = render({'models':[model]},{'Qwen3/0.6B':result})
    self.assertIn('w8a8 / FP16',table)
    self.assertIn('791.11 / 105.00*',table)
    self.assertNotIn('**105.00',table)
    with self.assertRaises(ValueError):render({'models':[model]},{'Qwen3/4B':result})

  def test_additional_model_without_reference_never_claims_a_win(self):
    model = {'model':'Spark-X2.6','size':'4B','dtype':None,'published_ttft_ms':None,
             'published_decode_tps':None,'published_memory_mb':None}
    coverage = {'models':[],'benchmark_additions':[model]}
    result = summarize(self.fixture())
    table = render(coverage,{'Spark-X2.6/4B':result},additional=True)
    self.assertIn('| Spark-X2.6 | 4B | — / FP16 | 128 | 64 | — / 204.00 | — / 14.00 | — / 105.00* |',table)
    self.assertNotIn('**',table)
    self.assertNotIn('Spark-X2.6',render(coverage,{'Spark-X2.6/4B':result}))
    with self.assertRaises(ValueError):render(coverage,{'Spark-X2.6/1.7B':result},additional=True)
    with self.assertRaises(ValueError):render({'models':[model],'benchmark_additions':[model]}, {})

  def test_prefill_proxy_units_and_absent_reference(self):
    published = {'model':'fixture','size':'small','dtype':'w8a8','published_ttft_ms':128.,
                 'published_decode_tps':10.,'published_memory_mb':100.}
    addition = {'model':'Spark-X2.6','size':'4B','dtype':None,'published_ttft_ms':None,
                'published_decode_tps':None,'published_memory_mb':None}
    result = summarize(self.fixture())
    table = render_prefill({'models':[published],'benchmark_additions':[addition]},
                           {'fixture/small':result,'Spark-X2.6/4B':result})
    self.assertIn('| fixture | small | **1000.00** / 627.45 |',table)
    self.assertIn('| Spark-X2.6 | 4B | — / 627.45 |',table)
