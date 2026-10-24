/* libncm_qnn: run QNN HTP context binaries in-process, with input/output buffers shared with the caller.
 *
 * qnn_serve.c talks over a pipe: every call copies the inputs into the server, then the outputs back.
 * This library is loaded by Python (ctypes) instead. Each graph tensor gets one buffer, allocated with
 * rpcmem (memory the DSP can map directly) and registered with QNN, so:
 *   - Python writes inputs straight into that buffer (a numpy view, no copy),
 *   - QNN hands the same memory to the NPU without staging copies,
 *   - Python reads outputs from the buffer the NPU wrote.
 * If rpcmem is unavailable (or NCM_QNN_RPCMEM=0), plain malloc buffers are used instead.
 *
 * API (all return 0 on success):
 *   ncm_open(backend_so, system_so, n, bins[])    load n context binaries (1 graph each)
 *   ncm_info(g, io, i, buf, len)                  "name bits scale offset rank d0 d1 ..." (io: 0 in, 1 out)
 *   ncm_count(g, io)                              number of tensors
 *   ncm_buffer(g, io, i)                          pointer to that tensor's buffer
 *   ncm_execute(g)                                run graph g on the buffers
 *   ncm_close()
 *
 * Build on the board:  gcc -O2 -shared -fPIC -I<qairt>/include/QNN qnn_lib.c -ldl -o libncm_qnn.so
 */
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"

#define ERR(...) do { fprintf(stderr, "libncm_qnn: " __VA_ARGS__); fputc('\n', stderr); return -1; } while (0)
#define MAXG 8
#define MAXT 8

typedef struct {
  Qnn_ContextHandle_t ctx;
  Qnn_GraphHandle_t h;
  uint32_t nin, nout;
  Qnn_Tensor_t t[2][MAXT];  /* [0] inputs, [1] outputs */
  size_t bytes[2][MAXT];
  void* buf[2][MAXT];
  int fd[2][MAXT];
} Graph;

static QNN_INTERFACE_VER_TYPE q;
static Qnn_BackendHandle_t backend;
static Qnn_DeviceHandle_t device;
static Graph G[MAXG];
static int ng;

/* rpcmem from libcdsprpc (loaded at run time so the library still works without it) */
#define RPCMEM_HEAP_ID_SYSTEM 25
#define RPCMEM_DEFAULT_FLAGS 1
static void* (*rpc_alloc)(int, uint32_t, int);
static void (*rpc_free)(void*);
static int (*rpc_to_fd)(void*);

static int dtype_bits(Qnn_DataType_t t) { return ((t >> 4) & 0xF) * 10 + (t & 0xF); }

static void load_rpcmem(void) {
  const char* e = getenv("NCM_QNN_RPCMEM");
  if (e && e[0] == '0') return;
  void* l = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_GLOBAL);
  if (!l) l = dlopen("libcdsprpc.so.1", RTLD_NOW | RTLD_GLOBAL);
  if (!l) return;
  rpc_alloc = dlsym(l, "rpcmem_alloc");
  rpc_free = dlsym(l, "rpcmem_free");
  rpc_to_fd = dlsym(l, "rpcmem_to_fd");
  if (!rpc_alloc || !rpc_free || !rpc_to_fd) rpc_alloc = NULL;
}

static void* load_file(const char* path, size_t* n) {
  FILE* f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  *n = ftell(f);
  fseek(f, 0, SEEK_SET);
  void* b = malloc(*n);
  if (fread(b, 1, *n, f) != *n) { free(b); b = NULL; }
  fclose(f);
  return b;
}

/* Copy one tensor description out of the binary info and give it a buffer. */
static int setup_tensor(Graph* gr, int io, int i, const Qnn_Tensor_t* src) {
  Qnn_Tensor_t* dst = &gr->t[io][i];
  *dst = *src;
  Qnn_TensorV1_t* d = &dst->v1; /* V2 starts with the same fields as V1 */
  d->name = strdup(src->v1.name);
  d->dimensions = malloc(sizeof(uint32_t) * d->rank);
  memcpy(d->dimensions, src->v1.dimensions, sizeof(uint32_t) * d->rank);
  if (dst->version == QNN_TENSOR_VERSION_2) {
    dst->v2.isDynamicDimensions = NULL;
    memset(&dst->v2.sparseParams, 0, sizeof(dst->v2.sparseParams));
  }
  size_t n = dtype_bits(d->dataType) / 8;
  for (uint32_t r = 0; r < d->rank; r++) n *= d->dimensions[r];
  gr->bytes[io][i] = n;
  gr->fd[io][i] = -1;

  if (rpc_alloc) {
    void* p = rpc_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, (int)n);
    int fd = p ? rpc_to_fd(p) : -1;
    if (p && fd >= 0) {
      Qnn_MemDescriptor_t md = QNN_MEM_DESCRIPTOR_INIT;
      md.memShape.numDim = d->rank;
      md.memShape.dimSize = d->dimensions;
      md.dataType = d->dataType;
      md.memType = QNN_MEM_TYPE_ION;
      md.ionInfo.fd = fd;
      Qnn_MemHandle_t mh = NULL;
      if (q.memRegister(gr->ctx, &md, 1, &mh) == QNN_SUCCESS) {
        d->memType = QNN_TENSORMEMTYPE_MEMHANDLE;
        d->memHandle = mh;
        gr->buf[io][i] = p;
        gr->fd[io][i] = fd;
        return 0;
      }
      fprintf(stderr, "libncm_qnn: memRegister failed for %s, using a normal buffer\n", d->name);
    }
    if (p) rpc_free(p);
  }
  d->memType = QNN_TENSORMEMTYPE_RAW;
  d->clientBuf.data = gr->buf[io][i] = calloc(1, n);
  d->clientBuf.dataSize = (uint32_t)n;
  return 0;
}

int ncm_open(const char* backend_so, const char* system_so, int n, const char** bins) {
  if (n > MAXG) ERR("too many graphs");
  void* lb = dlopen(backend_so, RTLD_NOW | RTLD_LOCAL);
  void* ls = dlopen(system_so, RTLD_NOW | RTLD_LOCAL);
  if (!lb || !ls) ERR("dlopen: %s", dlerror());
  typedef Qnn_ErrorHandle_t (*GetP)(const QnnInterface_t***, uint32_t*);
  typedef Qnn_ErrorHandle_t (*GetSP)(const QnnSystemInterface_t***, uint32_t*);
  const QnnInterface_t** ps;
  const QnnSystemInterface_t** sps;
  uint32_t np, nsp;
  if (((GetP)dlsym(lb, "QnnInterface_getProviders"))(&ps, &np) || np == 0) ERR("no QNN provider");
  if (((GetSP)dlsym(ls, "QnnSystemInterface_getProviders"))(&sps, &nsp) || nsp == 0) ERR("no QNN system provider");
  q = ps[0]->QNN_INTERFACE_VER_NAME;
  QNN_SYSTEM_INTERFACE_VER_TYPE sq = sps[0]->QNN_SYSTEM_INTERFACE_VER_NAME;
  if (q.backendCreate(NULL, NULL, &backend)) ERR("backendCreate failed");
  if (q.deviceCreate && q.deviceCreate(NULL, NULL, &device)) ERR("deviceCreate failed");
  load_rpcmem();

  for (int g = 0; g < n; g++) {
    Graph* gr = &G[g];
    size_t sz;
    void* buf = load_file(bins[g], &sz);
    if (!buf) ERR("cannot read %s", bins[g]);
    QnnSystemContext_Handle_t sh;
    const QnnSystemContext_BinaryInfo_t* bi;
    Qnn_ContextBinarySize_t bis;
    if (sq.systemContextCreate(&sh)) ERR("systemContextCreate failed");
    if (sq.systemContextGetBinaryInfo(sh, buf, sz, &bi, &bis)) ERR("getBinaryInfo failed: %s", bins[g]);
    uint32_t ngr;
    QnnSystemContext_GraphInfo_t* gi;
    if (bi->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) { ngr = bi->contextBinaryInfoV1.numGraphs; gi = bi->contextBinaryInfoV1.graphs; }
    else if (bi->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) { ngr = bi->contextBinaryInfoV2.numGraphs; gi = bi->contextBinaryInfoV2.graphs; }
    else { ngr = bi->contextBinaryInfoV3.numGraphs; gi = bi->contextBinaryInfoV3.graphs; }
    if (ngr != 1) ERR("%s: expected 1 graph, found %u", bins[g], ngr);
    const QnnSystemContext_GraphInfoV1_t* gv = &gi[0].graphInfoV1; /* V2/V3 share these leading fields */
    gr->nin = gv->numGraphInputs;
    gr->nout = gv->numGraphOutputs;
    if (gr->nin > MAXT || gr->nout > MAXT) ERR("too many tensors");
    if (q.contextCreateFromBinary(backend, device, NULL, buf, sz, &gr->ctx, NULL)) ERR("contextCreateFromBinary failed: %s", bins[g]);
    if (q.graphRetrieve(gr->ctx, gv->graphName, &gr->h)) ERR("graphRetrieve failed: %s", gv->graphName);
    for (uint32_t i = 0; i < gr->nin; i++) setup_tensor(gr, 0, i, &gv->graphInputs[i]);
    for (uint32_t i = 0; i < gr->nout; i++) setup_tensor(gr, 1, i, &gv->graphOutputs[i]);
    sq.systemContextFree(sh);
    free(buf);
  }
  ng = n;
  return 0;
}

int ncm_count(int g, int io) { return io ? G[g].nout : G[g].nin; }

void* ncm_buffer(int g, int io, int i) { return G[g].buf[io][i]; }

int ncm_shared(int g, int io, int i) { return G[g].fd[io][i] >= 0; }

int ncm_info(int g, int io, int i, char* out, int len) {
  const Qnn_TensorV1_t* v = &G[g].t[io][i].v1;
  float s = 1.f;
  int32_t o = 0;
  if (v->quantizeParams.encodingDefinition == QNN_DEFINITION_DEFINED &&
      v->quantizeParams.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
    s = v->quantizeParams.scaleOffsetEncoding.scale;
    o = v->quantizeParams.scaleOffsetEncoding.offset;
  }
  /* bits; signed/float flag: QNN dtype high nibble 0=int 1=uint 2=float 3/4=(u)fixed point */
  int k = snprintf(out, len, "%s %d %d %.9g %d %u", v->name, dtype_bits(v->dataType), (v->dataType >> 8) & 0xF, s, o, v->rank);
  for (uint32_t r = 0; r < v->rank && k < len; r++) k += snprintf(out + k, len - k, " %u", v->dimensions[r]);
  return 0;
}

int ncm_execute(int g) {
  Graph* gr = &G[g];
  if (q.graphExecute(gr->h, gr->t[0], gr->nin, gr->t[1], gr->nout, NULL, NULL)) ERR("graphExecute failed (graph %d)", g);
  return 0;
}

void ncm_close(void) {
  for (int g = 0; g < ng; g++) {
    for (int io = 0; io < 2; io++)
      for (uint32_t i = 0; i < (io ? G[g].nout : G[g].nin); i++) {
        if (G[g].fd[io][i] >= 0) {
          q.memDeRegister(&G[g].t[io][i].v1.memHandle, 1);
          rpc_free(G[g].buf[io][i]);
        } else {
          free(G[g].buf[io][i]);
        }
      }
    q.contextFree(G[g].ctx, NULL);
  }
  if (device && q.deviceFree) q.deviceFree(device);
  q.backendFree(backend);
  ng = 0;
}
