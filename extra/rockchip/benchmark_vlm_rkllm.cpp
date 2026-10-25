// Full vendor image encoder + connector + image-conditioned text generation.
// Run under flock and sdk_entry.py so custom-backend state is reset first.
#include <cstddef>
#include "rkllm.h"
#include "rknn_api.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

using Clock=std::chrono::steady_clock;
static std::vector<int> generated;
static Clock::time_point first,last;
static bool failed;
static RKLLMPerfStat perf;
static int callback(RKLLMResult *result,void *,LLMCallState state) {
  if(state==RKLLM_RUN_ERROR)failed=true;
  if(state==RKLLM_RUN_NORMAL) {
    if(generated.empty())first=Clock::now();
    last=Clock::now();generated.push_back(result->token_id);
  }
  if(state==RKLLM_RUN_FINISH)perf=result->perf;
  return 0;
}
static void check(int status,const char *what) {if(status)throw std::runtime_error(std::string(what)+": "+std::to_string(status));}
static std::vector<char> read_bytes(const std::string &path) {
  std::ifstream file(path,std::ios::binary);
  if(!file)throw std::runtime_error("Cannot read "+path);
  return std::vector<char>(std::istreambuf_iterator<char>(file),{});
}
static std::vector<float> read_floats(const std::string &path) {
  auto bytes=read_bytes(path);size_t offset=0;
  if(bytes.size()>10 && !std::memcmp(bytes.data(),"\x93NUMPY",6)) {
    if(bytes[6]!=1)throw std::runtime_error("Need NPY version 1");
    offset=10+(unsigned char)bytes[8]+256*(unsigned char)bytes[9];
  }
  if((bytes.size()-offset)%4)throw std::runtime_error("Misaligned float input");
  std::vector<float> result((bytes.size()-offset)/4);
  std::memcpy(result.data(),bytes.data()+offset,bytes.size()-offset);
  return result;
}
static double ms(Clock::time_point a,Clock::time_point b) {return std::chrono::duration<double,std::milli>(b-a).count();}

int main(int argc,char **argv) {
  try {
    if(argc<7 || argc>9)return 2;
    std::string fixture=argv[3];int tokens=std::atoi(argv[4]),warmups=std::atoi(argv[5]),runs=std::atoi(argv[6]);
    int context=argc>=8?std::atoi(argv[7]):512,warmup_rows=argc==9?std::atoi(argv[8]):0;
    auto pixels=read_floats(fixture+"/pixels.npy"),base=read_floats(fixture+"/input_embeddings.f32");
    if(pixels.size()!=3*512*512 || base.size()%576 || tokens<2 || warmups<3 || runs<1)return 2;
    size_t count=base.size()/576;
    if(context<1 || count+tokens>(size_t)context || warmup_rows<0 || (warmup_rows && (warmup_rows<128 || warmup_rows>(int)count)))return 2;
    auto json=read_bytes(fixture+"/input_ids.json");
    std::string digits(json.begin(),json.end());
    std::vector<int> ids;const char *cursor=digits.c_str();
    while(*cursor) {if(*cursor>='0'&&*cursor<='9') {char *end;ids.push_back((int)std::strtol(cursor,&end,10));cursor=end;}else cursor++;}
    if(ids.size()!=count || std::count(ids.begin(),ids.end(),49190)!=64)return 2;
    rknn_context encoder=0;auto vision_file=read_bytes(argv[1]);
    check(rknn_init(&encoder,vision_file.data(),vision_file.size(),0,nullptr),"rknn_init");
    check(rknn_set_core_mask(encoder,RKNN_NPU_CORE_0_1_2),"rknn_set_core_mask");
    rknn_tensor_attr input_attr{};input_attr.index=0;
    check(rknn_query(encoder,RKNN_QUERY_INPUT_ATTR,&input_attr,sizeof(input_attr)),"rknn_query_input");
    std::printf("{\"event\":\"input_contract\",\"type\":%d,\"format\":%d,\"dimensions\":%d,\"elements\":%d}\n",
                (int)input_attr.type,(int)input_attr.fmt,input_attr.n_dims,input_attr.n_elems);std::fflush(stdout);
    rknn_tensor_format input_format=RKNN_TENSOR_NCHW;
    if(input_attr.fmt==RKNN_TENSOR_NHWC) {
      auto nchw=pixels;
      for(int y=0;y<512;y++)for(int x=0;x<512;x++)for(int c=0;c<3;c++)
        pixels[(y*512+x)*3+c]=nchw[c*512*512+y*512+x];
      input_format=RKNN_TENSOR_NHWC;
    }
    std::printf("{\"event\":\"pixel_range\",\"min\":%.6f,\"max\":%.6f}\n",
                *std::min_element(pixels.begin(),pixels.end()),*std::max_element(pixels.begin(),pixels.end()));std::fflush(stdout);
    rknn_tensor_attr output_attr{};output_attr.index=0;
    check(rknn_query(encoder,RKNN_QUERY_OUTPUT_ATTR,&output_attr,sizeof(output_attr)),"rknn_query");
    if(std::string(argv[2])!="-" && output_attr.n_elems!=64*576)throw std::runtime_error("Unexpected image feature size");
    if(std::string(argv[2])=="-") {
      for(int index=0;index<warmups+runs;index++) {
        auto begin=Clock::now();
        rknn_input input{};input.index=0;input.buf=pixels.data();input.size=pixels.size()*4;
        input.type=RKNN_TENSOR_FLOAT32;input.fmt=input_format;
        check(rknn_inputs_set(encoder,1,&input),"rknn_inputs_set");
        check(rknn_run(encoder,nullptr),"rknn_run");
        rknn_output output{};output.index=0;output.want_float=1;
        check(rknn_outputs_get(encoder,1,&output,nullptr),"rknn_outputs_get");
        auto end=Clock::now();
        if(output.size!=output_attr.n_elems*4)throw std::runtime_error("Unexpected output bytes");
        if(index==0) {
          auto file=std::fopen((fixture+"/vendor_image_embeddings.f32").c_str(),"wb");
          if(!file || std::fwrite(output.buf,1,output.size,file)!=output.size)throw std::runtime_error("Cannot write image features");
          std::fclose(file);
        }
        check(rknn_outputs_release(encoder,1,&output),"rknn_outputs_release");
        std::printf("{\"event\":\"vision_only\",\"index\":%d,\"warmup\":%s,\"vision_ms\":%.6f}\n",
                    index,index<warmups?"true":"false",ms(begin,end));std::fflush(stdout);
      }
      check(rknn_destroy(encoder),"rknn_destroy");return 0;
    }
    auto param=rkllm_createDefaultParam();
    param.model_path=argv[2];param.max_context_len=context;param.max_new_tokens=tokens;
    param.top_k=1;param.top_p=1;param.temperature=0;param.repeat_penalty=1;
    param.frequency_penalty=param.presence_penalty=0;param.ignore_eos_token=true;param.skip_special_token=false;
    param.extend_param.enabled_cpus_num=4;param.extend_param.enabled_cpus_mask=0xf0;
    param.extend_param.embed_flash=1;param.extend_param.base_domain_id=1;param.extend_param.n_batch=1;
    RKLLMCallback cb{};cb.result_callback=callback;LLMHandle decoder=nullptr;
    check(rkllm_init(&decoder,&param,&cb),"rkllm_init");
    check(rkllm_set_chat_template(decoder,"","",""),"rkllm_set_chat_template");
    for(int index=0;index<warmups+runs;index++) {
      check(rkllm_clear_kv_cache(decoder,0,nullptr,nullptr),"rkllm_clear_kv_cache");
      generated.clear();failed=false;perf={};
      auto begin=Clock::now();
      rknn_input input{};input.index=0;input.buf=pixels.data();input.size=pixels.size()*4;
      input.type=RKNN_TENSOR_FLOAT32;input.fmt=input_format;input.pass_through=0;
      check(rknn_inputs_set(encoder,1,&input),"rknn_inputs_set");
      check(rknn_run(encoder,nullptr),"rknn_run");
      rknn_output output{};output.index=0;output.want_float=1;
      check(rknn_outputs_get(encoder,1,&output,nullptr),"rknn_outputs_get");
      auto *image=(float*)output.buf;
      if(output.size!=64*576*4)throw std::runtime_error("Unexpected output bytes");
      if(index==0) {
        auto file=std::fopen((fixture+"/vendor_image_embeddings.f32").c_str(),"wb");
        if(!file || std::fwrite(image,4,64*576,file)!=64*576)throw std::runtime_error("Cannot write image features");
        std::fclose(file);
      }
      auto encoded=Clock::now();auto embeddings=base;size_t slot=0;
      for(size_t i=0;i<count;i++)if(ids[i]==49190) {
        for(int d=0;d<576;d++) {
          float value=image[slot*576+d];
          if(!std::isfinite(value))throw std::runtime_error("Nonfinite image feature");
          embeddings[i*576+d]=value;
        }
        slot++;
      }
      check(rknn_outputs_release(encoder,1,&output),"rknn_outputs_release");
      RKLLMInput request{};request.input_type=RKLLM_INPUT_EMBED;request.role=nullptr;
      int input_count=index<warmups&&warmup_rows?warmup_rows:(int)count;
      if(std::count(ids.begin(),ids.begin()+input_count,49190)!=64)throw std::runtime_error("Warmup omits image tokens");
      request.embed_input.embed=embeddings.data();request.embed_input.n_tokens=input_count;
      RKLLMInferParam infer{};infer.mode=RKLLM_INFER_GENERATE;infer.keep_history=1;
      check(rkllm_run(decoder,&request,&infer,nullptr),"rkllm_run");
      auto end=Clock::now();
      if(failed || generated.size()!=(size_t)tokens)throw std::runtime_error("Incomplete vendor response");
      int cached=0;check(rkllm_get_kv_cache_size(decoder,&cached),"rkllm_get_kv_cache_size");
      if(perf.prefill_tokens!=input_count || cached<input_count || cached>input_count+tokens)
        throw std::runtime_error("Vendor truncated the prompt or cache");
      std::printf("{\"event\":\"trial\",\"index\":%d,\"warmup\":%s,\"vision_ms\":%.6f,\"first_token_ms\":%.6f,"
                  "\"prefill_tps\":%.6f,\"decode_tps\":%.6f,\"image_to_answer_ms\":%.6f,"
                  "\"sdk_prefill_ms\":%.6f,\"sdk_decode_ms\":%.6f,\"context\":%d,\"prompt_tokens\":%d,\"kv_tokens\":%d,\"generated_ids\":[",
                  index,index<warmups?"true":"false",ms(begin,encoded),ms(begin,first),
                  input_count*1000/ms(encoded,first),(tokens-1)*1000/ms(first,last),ms(begin,end),perf.prefill_time_ms,perf.generate_time_ms,
                  context,input_count,cached);
      for(size_t i=0;i<generated.size();i++)std::printf("%s%d",i?",":"",generated[i]);
      std::puts("]}");std::fflush(stdout);
    }
    check(rkllm_destroy(decoder),"rkllm_destroy");check(rknn_destroy(encoder),"rknn_destroy");return 0;
  } catch(const std::exception &error) {std::fprintf(stderr,"%s\n",error.what());return 1;}
}
