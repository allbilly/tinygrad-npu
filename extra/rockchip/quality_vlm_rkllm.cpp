// Image-conditioned teacher-forced logits and greedy answers using stock runtimes.
#include <cstddef>
#include "rkllm.h"
#include "rknn_api.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static std::vector<float> logits;
static bool failed;
static int callback(RKLLMResult *result,void *,LLMCallState state) {
  if(state==RKLLM_RUN_ERROR)failed=true;
  if(state!=RKLLM_RUN_NORMAL || !result->logits.logits)return 0;
  int tokens=result->logits.num_tokens,vocab=result->logits.vocab_size;
  if(tokens<1 || vocab!=49280) {
    std::fprintf(stderr,"Unexpected logits shape: tokens=%d vocab=%d\n",tokens,vocab);failed=true;return 0;
  }
  const float *last=result->logits.logits+(size_t)(tokens-1)*vocab;
  logits.assign(last,last+vocab);return 0;
}
static void check(int status,const char *what) {if(status)throw std::runtime_error(std::string(what)+": "+std::to_string(status));}
static std::vector<char> bytes(const std::string &path) {
  std::ifstream file(path,std::ios::binary);
  if(!file)throw std::runtime_error("Cannot read "+path);
  return std::vector<char>(std::istreambuf_iterator<char>(file),{});
}
static std::vector<float> floats(const std::string &path) {
  auto data=bytes(path);size_t offset=0;
  if(data.size()>10 && !std::memcmp(data.data(),"\x93NUMPY",6)) {
    if(data[6]!=1)throw std::runtime_error("Need NPY version 1");
    offset=10+(unsigned char)data[8]+256*(unsigned char)data[9];
  }
  if(offset>data.size() || (data.size()-offset)%4)throw std::runtime_error("Invalid float input");
  std::vector<float> output((data.size()-offset)/4);std::memcpy(output.data(),data.data()+offset,output.size()*4);return output;
}
static std::vector<int> ids(const std::string &path) {
  auto data=bytes(path);std::string value(data.begin(),data.end());std::vector<int> output;
  const char *cursor=value.c_str();
  while(*cursor) {if(*cursor>='0'&&*cursor<='9') {char *end;output.push_back(std::strtol(cursor,&end,10));cursor=end;}else cursor++;}
  return output;
}
static void dump(const std::string &path,const std::vector<float> &data) {
  auto file=std::fopen(path.c_str(),"wb");
  if(!file || std::fwrite(data.data(),4,data.size(),file)!=data.size())throw std::runtime_error("Cannot write "+path);
  std::fclose(file);
}
static void run(LLMHandle decoder,RKLLMInput &input,int expected_cache) {
  RKLLMInferParam infer{};infer.mode=RKLLM_INFER_GET_LOGITS;infer.keep_history=1;
  logits.clear();failed=false;check(rkllm_run(decoder,&input,&infer,nullptr),"rkllm_run");
  if(failed || logits.size()!=49280)throw std::runtime_error("Invalid logits callback; elements="+std::to_string(logits.size()));
  if(!std::all_of(logits.begin(),logits.end(),[](float value){return std::isfinite(value);})) {
    int nan=0,positive=0,negative=0;
    for(float value:logits) {nan+=std::isnan(value);positive+=value==INFINITY;negative+=value==-INFINITY;}
    throw std::runtime_error("Nonfinite logits: nan="+std::to_string(nan)+" posinf="+std::to_string(positive)+" neginf="+std::to_string(negative));
  }
  int cached=0;check(rkllm_get_kv_cache_size(decoder,&cached),"rkllm_get_kv_cache_size");
  if(cached!=expected_cache)throw std::runtime_error("Incorrect KV length: "+std::to_string(cached)+" expected "+std::to_string(expected_cache));
}

int main(int argc,char **argv) {
  try {
    if(argc!=5&&argc!=6)return 2;
    int context=argc==6?std::atoi(argv[5]):512;
    if(context<1)return 2;
    rknn_context encoder=0;auto artifact=bytes(argv[1]);
    check(rknn_init(&encoder,artifact.data(),artifact.size(),0,nullptr),"rknn_init");
    check(rknn_set_core_mask(encoder,RKNN_NPU_CORE_0_1_2),"rknn_set_core_mask");
    rknn_tensor_attr input_attr{};input_attr.index=0;
    check(rknn_query(encoder,RKNN_QUERY_INPUT_ATTR,&input_attr,sizeof(input_attr)),"rknn_query_input");
    auto param=rkllm_createDefaultParam();param.model_path=argv[2];param.max_context_len=context;param.max_new_tokens=1;
    param.top_k=1;param.top_p=1;param.temperature=0;param.repeat_penalty=1;
    param.frequency_penalty=param.presence_penalty=0;param.ignore_eos_token=false;param.skip_special_token=false;
    param.extend_param.enabled_cpus_num=4;param.extend_param.enabled_cpus_mask=0xf0;
    param.extend_param.embed_flash=1;param.extend_param.base_domain_id=1;param.extend_param.n_batch=1;
    RKLLMCallback cb{};cb.result_callback=callback;LLMHandle decoder=nullptr;
    check(rkllm_init(&decoder,&param,&cb),"rkllm_init");check(rkllm_set_chat_template(decoder,"","",""),"chat_template");
    std::ifstream manifest(argv[3]);std::string line;
    while(std::getline(manifest,line)) {
      std::istringstream row(line);std::string name,fixture;int positions,value;
      if(!(row>>name>>fixture>>positions) || name.find('/')!=std::string::npos || positions<1 || positions>16)return 2;
      std::vector<int> gold;while(row>>value)gold.push_back(value);
      if(gold.size()!=(size_t)positions)return 2;
      auto pixels=floats(fixture+"/pixels.npy"),embeddings=floats(fixture+"/input_embeddings.f32");
      auto input_ids=ids(fixture+"/input_ids.json");
      int prompt=input_ids.size();
      if(pixels.size()!=3*512*512 || embeddings.size()!=(size_t)prompt*576 || prompt+std::max(positions,16)>context)return 2;
      if(input_attr.fmt==RKNN_TENSOR_NHWC) {
        auto nchw=pixels;
        for(int y=0;y<512;y++)for(int x=0;x<512;x++)for(int c=0;c<3;c++)pixels[(y*512+x)*3+c]=nchw[c*512*512+y*512+x];
      }
      rknn_input image{};image.index=0;image.buf=pixels.data();image.size=pixels.size()*4;
      image.type=RKNN_TENSOR_FLOAT32;image.fmt=input_attr.fmt;
      check(rknn_inputs_set(encoder,1,&image),"rknn_inputs_set");check(rknn_run(encoder,nullptr),"rknn_run");
      rknn_output output{};output.index=0;output.want_float=1;
      check(rknn_outputs_get(encoder,1,&output,nullptr),"rknn_outputs_get");
      if(output.size!=64*576*4)throw std::runtime_error("Wrong image feature shape");
      auto *features=(float*)output.buf;int slot=0;
      for(int t=0;t<prompt;t++)if(input_ids[t]==49190) {
        if(slot>=64)throw std::runtime_error("Too many image tokens");
        std::copy(features+slot*576,features+(slot+1)*576,embeddings.begin()+t*576);slot++;
      }
      if(slot!=64)throw std::runtime_error("Missing image tokens");
      dump(std::string(argv[4])+"/"+name+".image.f32",std::vector<float>(features,features+64*576));
      check(rknn_outputs_release(encoder,1,&output),"rknn_outputs_release");
      RKLLMInput request{};request.input_type=RKLLM_INPUT_EMBED;request.role=nullptr;
      request.embed_input.embed=embeddings.data();request.embed_input.n_tokens=prompt;
      std::vector<float> all;
      check(rkllm_clear_kv_cache(decoder,0,nullptr,nullptr),"clear_kv");run(decoder,request,prompt);
      for(int step=0;step<positions;step++) {
        if(step) {
          int32_t token=gold[step-1];RKLLMInput next{};next.input_type=RKLLM_INPUT_TOKEN;
          next.token_input.input_ids=&token;next.token_input.n_tokens=1;run(decoder,next,prompt+step);
        }
        all.insert(all.end(),logits.begin(),logits.end());
      }
      dump(std::string(argv[4])+"/"+name+".logits.f32",all);
      // A separate empty-history pass gives the model's actual answer, rather
      // than treating teacher-forced argmaxes as a generated response.
      check(rkllm_clear_kv_cache(decoder,0,nullptr,nullptr),"clear_kv");run(decoder,request,prompt);
      std::vector<int> generated;
      for(int step=0;step<16;step++) {
        int32_t token=std::max_element(logits.begin(),logits.end())-logits.begin();generated.push_back(token);
        if(token==49279 || step==15)break;
        RKLLMInput next{};next.input_type=RKLLM_INPUT_TOKEN;next.token_input.input_ids=&token;next.token_input.n_tokens=1;
        run(decoder,next,prompt+step+1);
      }
      std::printf("{\"event\":\"case\",\"name\":\"%s\",\"positions\":%d,\"generated_ids\":[",name.c_str(),positions);
      for(size_t i=0;i<generated.size();i++)std::printf("%s%d",i?",":"",generated[i]);
      std::puts("]}");std::fflush(stdout);
    }
    check(rkllm_destroy(decoder),"rkllm_destroy");check(rknn_destroy(encoder),"rknn_destroy");return 0;
  }catch(const std::exception &error){std::fprintf(stderr,"%s\n",error.what());return 1;}
}
