"""Frozen Sentinel token encoder adapter. Caller supplies canonical 256x256 RGB floats."""
import hashlib, threading, time
from pathlib import Path
import numpy as np
from tflite_runtime.interpreter import Interpreter, OpResolverType, load_delegate

MODELS = {
 'maskgit': {'sha256':'0a418626321b95b96125d7aa61a07ce27f453cb06d042f23f146b9f59a3c4b2b','tokens':256,'codebook_size':1024},
 'resnet': {'sha256':'561ecd5aa6408fb72f48ae556654ad4a5b111b0e0a9d1979e1e17bc2e8cff25b','tokens':128,'codebook_size':4096},
}
def sha(data): return hashlib.sha256(data).hexdigest()
def describe(d):
 q=d['quantization_parameters']
 return {'name':d['name'],'index':int(d['index']),'shape':d['shape'].tolist(),'dtype':np.dtype(d['dtype']).name,'scale':q['scales'].tolist(),'zero_point':q['zero_points'].tolist(),'quantized_dimension':int(q['quantized_dimension']),'quantization':[float(d['quantization'][0]),int(d['quantization'][1])]}

class TokenEncoder:
 """One persistent interpreter; calls serialized and input reset before every invoke."""
 def __init__(self, model, kind='maskgit', delegate='/usr/lib/libethosu_delegate.so', threads=4, delegate_options=None):
  self.kind=kind;self.spec=MODELS[kind];self.model=Path(model);self.lock=threading.Lock();self.timings={}
  t=time.perf_counter();self.model_sha256=sha(self.model.read_bytes());self.timings['model_hash_read_ms']=(time.perf_counter()-t)*1000
  if self.model_sha256 != self.spec['sha256']: raise ValueError('Model digest differs from the validated frozen artifact')
  t=time.perf_counter();self.delegate=load_delegate(delegate,options=delegate_options or {});self.timings['delegate_load_ms']=(time.perf_counter()-t)*1000
  t=time.perf_counter();self.interpreter=Interpreter(model_path=str(self.model),num_threads=threads,experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,experimental_delegates=[self.delegate]);self.timings['interpreter_create_ms']=(time.perf_counter()-t)*1000
  t=time.perf_counter();self.interpreter.allocate_tensors();self.timings['allocate_tensors_ms']=(time.perf_counter()-t)*1000
  self.input=self.interpreter.get_input_details()[0];self.outputs=self.interpreter.get_output_details()
  if self.input['shape'].tolist()!=[1,256,256,3]:raise ValueError('Unexpected input shape')
  self.metadata={'kind':kind,'model_sha256':self.model_sha256,'inputs':[describe(self.input)],'outputs':[describe(d) for d in self.outputs],'setup_timing_ms':self.timings}

 def prepare(self, x):
  """No hidden resize/padding. x must be finite float32 NCHW RGB in [0,1]."""
  x=np.asarray(x)
  if x.dtype!=np.float32 or x.shape!=(1,3,256,256) or not np.isfinite(x).all() or x.min()<0 or x.max()>1:raise ValueError('Expected canonical float32 [1,3,256,256], finite and within [0,1]')
  t=time.perf_counter();y=x.transpose(0,2,3,1);s,z=self.input['quantization'];dt=self.input['dtype']
  if np.issubdtype(dt,np.integer):
   if s<=0:raise ValueError('Missing scalar input quantization')
   bounds=np.iinfo(dt);q=np.clip(np.rint(y/s+z),bounds.min,bounds.max).astype(dt)
  else:q=y.astype(dt)
  return q,{'prepare_ms':(time.perf_counter()-t)*1000,'input_float_nchw_sha256':sha(x.tobytes()),'input_quantized_sha256':sha(q.tobytes())}

 def run_prepared(self, q, capture_outputs=False):
  with self.lock:
   t=time.perf_counter();self.interpreter.set_tensor(self.input['index'],q);set_ms=(time.perf_counter()-t)*1000
   t=time.perf_counter();self.interpreter.invoke();invoke_ms=(time.perf_counter()-t)*1000
   t=time.perf_counter();raw={};ids=None
   if self.kind=='maskgit':
    d=next(d for d in self.outputs if d['shape'].tolist()==[1,16,16] and np.issubdtype(d['dtype'],np.integer));ids=self.interpreter.get_tensor(d['index']).reshape(-1)
   else:
    d=self.outputs[0];v=self.interpreter.get_tensor(d['index']);axis=[a for a,n in enumerate(v.shape) if n==4096]
    if len(axis)!=1 or len(d['quantization_parameters']['scales'])>1:raise ValueError('ResNet score ABI changed')
    ids=v.argmax(axis=axis[0]).reshape(-1)
   if len(ids)!=self.spec['tokens'] or ids.min()<0 or ids.max()>=self.spec['codebook_size']:raise ValueError('Token IDs outside decoder contract')
   payload=ids.astype('<u2').tobytes();post_ms=(time.perf_counter()-t)*1000
   t=time.perf_counter()
   if capture_outputs:raw={f'output_{i}':self.interpreter.get_tensor(d['index']).copy() for i,d in enumerate(self.outputs)}
   capture_ms=(time.perf_counter()-t)*1000
   return payload,raw,{'set_input_ms':set_ms,'invoke_ms':invoke_ms,'token_extract_pack_ms':post_ms,'capture_copy_ms':capture_ms,'token_payload_sha256':sha(payload),'token_payload_bytes':len(payload),'output_sha256':{k:sha(v.tobytes()) for k,v in raw.items()}}

 def encode(self, x):
  q,_=self.prepare(x);return self.run_prepared(q)[0]

def unpack_tokens(payload, kind='maskgit'):
 spec=MODELS[kind]
 if len(payload)!=2*spec['tokens']:raise ValueError('Token payload length differs from ABI')
 ids=np.frombuffer(payload,dtype='<u2').astype(np.int64)
 if ids.max()>=spec['codebook_size']:raise ValueError('Invalid codebook index')
 return ids.reshape(1,-1)
