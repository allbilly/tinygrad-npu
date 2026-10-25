// Raw-token cold-KV SDK measurements with verified full prompt consumption.
// Run with sdk_entry.py under the exclusive NPU submission lock.
#include <cstddef>
#include "rkllm.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

using Clock=std::chrono::steady_clock;
static Clock::time_point first,last;
static std::vector<int> generated;
static RKLLMPerfStat perf;
static bool failed;
static int callback(RKLLMResult *result,void *,LLMCallState state) {
  if(state==RKLLM_RUN_ERROR)failed=true;
  if(state==RKLLM_RUN_NORMAL) {
    if(generated.empty())first=Clock::now();
    last=Clock::now();generated.push_back(result->token_id);
  }
  if(state==RKLLM_RUN_FINISH)perf=result->perf;
  return 0;
}
static double ms(Clock::time_point a,Clock::time_point b) {
  return std::chrono::duration<double,std::milli>(b-a).count();
}

int main(int argc,char **argv) {
  if(argc!=7&&argc!=8)return 2;
  std::ifstream file(argv[2]);std::vector<int32_t> ids;int32_t id;
  while(file>>id)ids.push_back(id);
  int context=std::atoi(argv[3]),tokens=std::atoi(argv[4]),warmups=std::atoi(argv[5]),runs=std::atoi(argv[6]);
  int warmup_rows=argc==8?std::atoi(argv[7]):0;
  if(ids.empty()||tokens<2||warmups<0||runs<1||ids.size()+tokens>(size_t)context||warmup_rows<0||warmup_rows>(int)ids.size())return 2;
  auto param=rkllm_createDefaultParam();
  param.model_path=argv[1];param.max_context_len=context;param.max_new_tokens=tokens;
  param.top_k=1;param.top_p=1;param.temperature=0;param.repeat_penalty=1;
  param.frequency_penalty=param.presence_penalty=0;param.ignore_eos_token=true;param.skip_special_token=false;
  param.extend_param.enabled_cpus_num=4;param.extend_param.enabled_cpus_mask=0xf0;
  param.extend_param.embed_flash=1;param.extend_param.base_domain_id=1;param.extend_param.n_batch=1;
  LLMHandle handle=nullptr;RKLLMCallback cb{};cb.result_callback=callback;
  auto initialized=Clock::now();
  int status=rkllm_init(&handle,&param,&cb);
  if(status){std::fprintf(stderr,"rkllm_init failed: %d\n",status);return 1;}
  std::printf("{\"event\":\"initialized\",\"context\":%d,\"prompt_tokens\":%zu,\"init_ms\":%.3f}\n",
              context,ids.size(),ms(initialized,Clock::now()));std::fflush(stdout);
  if(rkllm_set_chat_template(handle,"","","")){rkllm_destroy(handle);return 1;}
  for(int index=0;index<warmups+runs;index++) {
    if(rkllm_clear_kv_cache(handle,0,nullptr,nullptr)){rkllm_destroy(handle);return 1;}
    RKLLMInput input{};input.role=nullptr;input.input_type=RKLLM_INPUT_TOKEN;
    int count=index<warmups&&warmup_rows?warmup_rows:(int)ids.size();
    input.token_input.input_ids=ids.data();input.token_input.n_tokens=count;
    RKLLMInferParam infer{};infer.mode=RKLLM_INFER_GENERATE;infer.keep_history=1;
    generated.clear();failed=false;perf={};auto start=Clock::now();
    status=rkllm_run(handle,&input,&infer,nullptr);auto end=Clock::now();int cache=0;
    int cache_status=rkllm_get_kv_cache_size(handle,&cache);
    bool consumed=perf.prefill_tokens==count&&cache>=count&&cache<=count+tokens;
    if(status||failed||generated.size()!=(size_t)tokens||cache_status||!consumed) {
      std::fprintf(stderr,"Invalid full-context run: status=%d failed=%d outputs=%zu/%d sdk_prefill=%d/%d cache=%d query=%d\n",
                   status,failed,generated.size(),tokens,perf.prefill_tokens,count,cache,cache_status);
      rkllm_destroy(handle);return 1;
    }
    std::printf("{\"event\":\"run\",\"warmup\":%s,\"index\":%d,\"context\":%d,\"prompt_tokens\":%d,"
      "\"new_tokens\":%d,\"kv_tokens\":%d,\"first_token_ms\":%.6f,\"prefill_tps\":%.6f,\"decode_tps\":%.6f,"
      "\"total_ms\":%.6f,\"sdk_prefill_tokens\":%d,\"sdk_prefill_ms\":%.6f,\"generated_ids\":[",
      index<warmups?"true":"false",index,context,count,tokens,cache,ms(start,first),count*1000/ms(start,first),
      (tokens-1)*1000/ms(first,last),ms(start,end),perf.prefill_tokens,perf.prefill_time_ms);
    for(size_t i=0;i<generated.size();i++)std::printf("%s%d",i?",":"",generated[i]);
    std::puts("]}");std::fflush(stdout);
  }
  rkllm_destroy(handle);return 0;
}
