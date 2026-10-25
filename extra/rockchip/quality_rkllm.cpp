// Dump full next-token logits for an explicit sequence of teacher-forced requests.
#include <cstddef>
#include "rkllm.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static std::string output_path;
static bool written, failed;
static int returned_tokens;
static bool want_logits=true;
static int expected_vocab=151936;
static int callback(RKLLMResult *result,void *,LLMCallState state) {
  if(state==RKLLM_RUN_ERROR) failed=true;
  if(state!=RKLLM_RUN_NORMAL || !want_logits || !result->logits.logits) return 0;
  int count=result->logits.num_tokens,vocab=result->logits.vocab_size;
  if(count<1 || vocab!=expected_vocab) {failed=true;return 0;}
  auto file=std::fopen(output_path.c_str(),"wb");
  const float *last=result->logits.logits+(size_t)(count-1)*vocab;
  if(!file || std::fwrite(last,sizeof(float),vocab,file)!=(size_t)vocab) failed=true;
  if(file)std::fclose(file);
  written=!failed;
  returned_tokens=count;
  return 0;
}

int main(int argc,char **argv) {
  if(argc<4||argc>8)return 2;
  int context=argc>=5?std::atoi(argv[4]):512;
  if(argc>=6&&std::string(argv[5])!="seed-hidden"&&std::string(argv[5])!="seed-generate")return 2;
  bool seed_generate=argc>=6&&std::string(argv[5])=="seed-generate";
  expected_vocab=argc>=7?std::atoi(argv[6]):151936;
  if(expected_vocab<1 || expected_vocab>1048576)return 2;
  if(context<1)return 2;
  auto param=rkllm_createDefaultParam();
  param.model_path=argv[1];param.max_context_len=context;param.max_new_tokens=1;
  param.top_k=1;param.top_p=1;param.temperature=0;param.repeat_penalty=1;
  param.frequency_penalty=param.presence_penalty=0;param.ignore_eos_token=true;param.skip_special_token=false;
  param.extend_param.enabled_cpus_num=4;param.extend_param.enabled_cpus_mask=0xf0;
  param.extend_param.embed_flash=1;param.extend_param.base_domain_id=1;param.extend_param.n_batch=1;
  RKLLMCallback cb{};cb.result_callback=callback;LLMHandle handle=nullptr;
  if(rkllm_init(&handle,&param,&cb))return 1;
  if(rkllm_set_chat_template(handle,"","",""))return 1;
  std::ifstream manifest(argv[2]);std::string line;int cached=0;
  while(std::getline(manifest,line)) {
    std::istringstream row(line);std::string name;int start,gold,value;
    if(!(row>>name>>start>>gold) || name.find('/')!=std::string::npos)return 2;
    std::vector<int32_t> ids;while(row>>value)ids.push_back(value);
    if(ids.empty() || start<0 || start+(int)ids.size()>context)return 2;
    if(start==0) {if(rkllm_clear_kv_cache(handle,0,nullptr,nullptr))return 1;cached=0;}
    if(cached!=start)return 2;
    RKLLMInput input{};input.role=nullptr;input.input_type=RKLLM_INPUT_TOKEN;
    input.token_input.input_ids=ids.data();input.token_input.n_tokens=ids.size();
    bool seed=argc>=6&&name=="seed";
    std::vector<float> embeddings;
    if(seed&&argc>=8&&std::string(argv[7])!="-") {
      std::ifstream file(argv[7],std::ios::binary|std::ios::ate);
      auto length=file.tellg();
      if(length!=(std::streamoff)(ids.size()*576*4))return 2;
      embeddings.resize(ids.size()*576);file.seekg(0);
      if(!file.read((char*)embeddings.data(),length))return 2;
      input.input_type=RKLLM_INPUT_EMBED;input.embed_input.embed=embeddings.data();input.embed_input.n_tokens=ids.size();
    }
    RKLLMInferParam infer{};
    infer.mode=seed?(seed_generate?RKLLM_INFER_GENERATE:RKLLM_INFER_GET_LAST_HIDDEN_LAYER):RKLLM_INFER_GET_LOGITS;infer.keep_history=1;
    want_logits=!seed;
    output_path=std::string(argv[3])+"/"+name+".f32";written=failed=false;returned_tokens=0;
    int error=rkllm_run(handle,&input,&infer,nullptr);
    if(error || failed || (!seed&&!written)) {rkllm_destroy(handle);return 1;}
    cached+=(int)ids.size();
    int actual_cache=0;
    if(rkllm_get_kv_cache_size(handle,&actual_cache) || actual_cache!=cached) {
      std::fprintf(stderr,"Unexpected KV length: actual=%d expected=%d\n",actual_cache,cached);
      rkllm_destroy(handle);return 1;
    }
    std::printf("{\"event\":\"%s\",\"name\":\"%s\",\"start\":%d,\"input_tokens\":%zu,\"returned_tokens\":%d,\"vocab\":%d}\n",
                seed?(seed_generate?"seed_generate":"hidden"):"logits",name.c_str(),start,ids.size(),returned_tokens,expected_vocab);
    std::fflush(stdout);
  }
  rkllm_destroy(handle);return 0;
}
