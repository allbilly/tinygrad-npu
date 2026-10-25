"""Measure a ready app's image prefill with greedy and sampled first tokens."""
import argparse, base64, hashlib, json, statistics, time, urllib.request
from pathlib import Path


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--url',default='http://127.0.0.1:8091')
  parser.add_argument('--model',default='b4b92c783d8643a1')
  parser.add_argument('--image',type=Path,required=True)
  parser.add_argument('--report',type=Path,required=True)
  parser.add_argument('--runs',type=int,default=5)
  args = parser.parse_args()
  client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
  def request(path,data=None):
    body = None if data is None else json.dumps(data).encode()
    with client.open(urllib.request.Request(args.url+path,body,{'Content-Type':'application/json'}),timeout=300) as response:
      return json.load(response)
  encoded = base64.b64encode(args.image.read_bytes()).decode()
  payload = {'model':args.model,'messages':[{'role':'user','content':[
    {'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+encoded}},
    {'type':'text','text':'What is shown in this image?'}]}],'max_tokens':2}
  rows = []
  # Alternate temperature order to reduce systematic thermal/order bias.
  for iteration in range(args.runs+2):
    for temperature in ((0.,.6) if iteration%2==0 else (.6,0.)):
      started = time.perf_counter()
      body = request('/v1/chat/completions',{**payload,'temperature':temperature})
      state = next(s for s in request('/api/status')['runtime']['sessions'] if s['model_id']==args.model)
      assert state['state']=='ready' and state['last_metrics']['prompt_tokens']==82,state
      rows.append({'iteration':iteration,'measured':iteration>=2,'temperature':temperature,
                   'wall_seconds':time.perf_counter()-started,'metrics':state['last_metrics'],
                   'answer':body['choices'][0]['message']['content']})
      print(json.dumps(rows[-1]),flush=True)
  medians = {str(t):{key:statistics.median(row['metrics'][key] for row in rows if row['measured'] and row['temperature']==t)
                      for key in ('prefill_ms','prefill_tps','vision_ms','ttft_ms')}
             for t in (0.,.6)}
  args.report.write_text(json.dumps({'complete':True,'model':args.model,'image_sha256':hashlib.sha256(args.image.read_bytes()).hexdigest(),
                                    'medians':medians,'rows':rows},indent=2)+'\n')
  print(json.dumps({'medians':medians}),flush=True)


if __name__=='__main__':main()
