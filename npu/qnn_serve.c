/* qnn_serve: keep QNN HTP context binaries loaded and run them on request.
 *
 *   qnn_serve libQnnHtp.so libQnnSystem.so a.bin [b.bin ...]
 *
 * On start it prints one text line per graph tensor, then "READY":
 *   T <graph#> <in|out> <tensor#> <name> <bits> <scale> <offset> <rank> <d0> <d1> ...
 * Then it loops on stdin: one byte = graph index (0xFF = quit), followed by each input's raw
 * bytes (as quantized by the caller: q = round(x/scale) - offset). It executes the graph and
 * writes every output's raw bytes to stdout (x = scale * (q + offset)).
 *
 * Build on the board:  gcc -O2 -I<qairt>/include/QNN qnn_serve.c -ldl -o qnn_serve
 */
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"

#define DIE(...) do { fprintf(stderr, "qnn_serve: " __VA_ARGS__); fputc('\n', stderr); exit(1); } while (0)
#define MAXG 8
#define MAXT 8

typedef struct {
  Qnn_GraphHandle_t h;
  uint32_t nin, nout;
  Qnn_Tensor_t in[MAXT], out[MAXT];
  size_t inb[MAXT], outb[MAXT];
} Graph;

static int dtype_bits(Qnn_DataType_t t) { return ((t >> 4) & 0xF) * 10 + (t & 0xF); }

static size_t tensor_bytes(const Qnn_TensorV1_t* t) {
  size_t n = 1;
  for (uint32_t i = 0; i < t->rank; i++) n *= t->dimensions[i];
  return n * (dtype_bits(t->dataType) / 8);
}

/* Deep-copy what we need from the binary info (it is freed with the system context). */
static void copy_tensor(Qnn_Tensor_t* dst, const Qnn_Tensor_t* src, size_t* bytes) {
  *dst = *src;
  Qnn_TensorV1_t* d = &dst->v1; /* V2 starts with the same fields as V1 */
  d->name = strdup(src->v1.name);
  d->dimensions = malloc(sizeof(uint32_t) * d->rank);
  memcpy(d->dimensions, src->v1.dimensions, sizeof(uint32_t) * d->rank);
  if (dst->version == QNN_TENSOR_VERSION_2) {
    dst->v2.isDynamicDimensions = NULL;
    memset(&dst->v2.sparseParams, 0, sizeof(dst->v2.sparseParams));
  }
  *bytes = tensor_bytes(d);
  d->memType = QNN_TENSORMEMTYPE_RAW;
  d->clientBuf.data = malloc(*bytes);
  d->clientBuf.dataSize = (uint32_t)*bytes;
}

static void print_tensor(int g, const char* io, int i, const Qnn_Tensor_t* t) {
  const Qnn_TensorV1_t* v = &t->v1;
  float s = 1.f;
  int32_t o = 0;
  if (v->quantizeParams.encodingDefinition == QNN_DEFINITION_DEFINED &&
      v->quantizeParams.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
    s = v->quantizeParams.scaleOffsetEncoding.scale;
    o = v->quantizeParams.scaleOffsetEncoding.offset;
  }
  printf("T %d %s %d %s %d %.9g %d %u", g, io, i, v->name, dtype_bits(v->dataType), s, o, v->rank);
  for (uint32_t r = 0; r < v->rank; r++) printf(" %u", v->dimensions[r]);
  printf("\n");
}

static void* load(const char* path, size_t* n) {
  FILE* f = fopen(path, "rb");
  if (!f) DIE("cannot open %s", path);
  fseek(f, 0, SEEK_END);
  *n = ftell(f);
  fseek(f, 0, SEEK_SET);
  void* b = malloc(*n);
  if (fread(b, 1, *n, f) != *n) DIE("short read %s", path);
  fclose(f);
  return b;
}

static void read_all(void* p, size_t n) {
  size_t got = 0;
  while (got < n) {
    size_t r = fread((char*)p + got, 1, n - got, stdin);
    if (r == 0) exit(0); /* caller went away */
    got += r;
  }
}

int main(int argc, char** argv) {
  if (argc < 4) DIE("usage: qnn_serve libQnnHtp.so libQnnSystem.so ctx.bin [...]");
  int ng = argc - 3;
  if (ng > MAXG) DIE("too many graphs");

  void* lb = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  void* ls = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
  if (!lb || !ls) DIE("dlopen: %s", dlerror());

  typedef Qnn_ErrorHandle_t (*GetP)(const QnnInterface_t***, uint32_t*);
  typedef Qnn_ErrorHandle_t (*GetSP)(const QnnSystemInterface_t***, uint32_t*);
  const QnnInterface_t** ps;
  const QnnSystemInterface_t** sps;
  uint32_t np, nsp;
  if (((GetP)dlsym(lb, "QnnInterface_getProviders"))(&ps, &np) || np == 0) DIE("no QNN provider");
  if (((GetSP)dlsym(ls, "QnnSystemInterface_getProviders"))(&sps, &nsp) || nsp == 0) DIE("no QNN system provider");
  QNN_INTERFACE_VER_TYPE q = ps[0]->QNN_INTERFACE_VER_NAME;
  QNN_SYSTEM_INTERFACE_VER_TYPE sq = sps[0]->QNN_SYSTEM_INTERFACE_VER_NAME;

  Qnn_BackendHandle_t backend;
  Qnn_DeviceHandle_t device = NULL;
  if (q.backendCreate(NULL, NULL, &backend)) DIE("backendCreate failed");
  if (q.deviceCreate && q.deviceCreate(NULL, NULL, &device)) DIE("deviceCreate failed");

  Graph G[MAXG];
  for (int g = 0; g < ng; g++) {
    size_t n;
    void* buf = load(argv[3 + g], &n);
    QnnSystemContext_Handle_t sh;
    const QnnSystemContext_BinaryInfo_t* bi;
    Qnn_ContextBinarySize_t bis;
    if (sq.systemContextCreate(&sh)) DIE("systemContextCreate failed");
    if (sq.systemContextGetBinaryInfo(sh, buf, n, &bi, &bis)) DIE("getBinaryInfo failed: %s", argv[3 + g]);
    uint32_t ngr;
    QnnSystemContext_GraphInfo_t* gi;
    if (bi->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) { ngr = bi->contextBinaryInfoV1.numGraphs; gi = bi->contextBinaryInfoV1.graphs; }
    else if (bi->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) { ngr = bi->contextBinaryInfoV2.numGraphs; gi = bi->contextBinaryInfoV2.graphs; }
    else { ngr = bi->contextBinaryInfoV3.numGraphs; gi = bi->contextBinaryInfoV3.graphs; }
    if (ngr != 1) DIE("%s: expected 1 graph, found %u", argv[3 + g], ngr);
    const QnnSystemContext_GraphInfoV1_t* gv = &gi[0].graphInfoV1; /* V2/V3 share these leading fields */
    G[g].nin = gv->numGraphInputs;
    G[g].nout = gv->numGraphOutputs;
    if (G[g].nin > MAXT || G[g].nout > MAXT) DIE("too many tensors");
    for (uint32_t i = 0; i < G[g].nin; i++) copy_tensor(&G[g].in[i], &gv->graphInputs[i], &G[g].inb[i]);
    for (uint32_t i = 0; i < G[g].nout; i++) copy_tensor(&G[g].out[i], &gv->graphOutputs[i], &G[g].outb[i]);
    char* gname = strdup(gv->graphName);
    sq.systemContextFree(sh);

    Qnn_ContextHandle_t ctx;
    if (q.contextCreateFromBinary(backend, device, NULL, buf, n, &ctx, NULL)) DIE("contextCreateFromBinary failed: %s", argv[3 + g]);
    if (q.graphRetrieve(ctx, gname, &G[g].h)) DIE("graphRetrieve failed: %s", gname);
    free(buf);
    for (uint32_t i = 0; i < G[g].nin; i++) print_tensor(g, "in", i, &G[g].in[i]);
    for (uint32_t i = 0; i < G[g].nout; i++) print_tensor(g, "out", i, &G[g].out[i]);
  }
  printf("READY\n");
  fflush(stdout);

  for (;;) {
    uint8_t g;
    read_all(&g, 1);
    if (g == 0xFF) break;
    if (g >= ng) DIE("bad graph index %u", g);
    for (uint32_t i = 0; i < G[g].nin; i++) read_all(G[g].in[i].v1.clientBuf.data, G[g].inb[i]);
    if (q.graphExecute(G[g].h, G[g].in, G[g].nin, G[g].out, G[g].nout, NULL, NULL)) DIE("graphExecute failed");
    for (uint32_t i = 0; i < G[g].nout; i++) fwrite(G[g].out[i].v1.clientBuf.data, 1, G[g].outb[i], stdout);
    fflush(stdout);
  }
  return 0;
}
