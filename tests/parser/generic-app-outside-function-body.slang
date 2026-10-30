//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-cpu -output-using-type

// Outside function bodies the parser has no semantic information, so it can only guess whether
// a `<` starts generic arguments. Such an expression is parsed again during checking, where `<`
// is decided like in a function body: `a < b || c > (d)` is two comparisons, not `a<...>(d)`.

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

int k<let N : int>() { return N; }

static const int a = 1, b = 2, c = 3, d = 4;
static const bool global = a < b || c > (d);
static const int tern = a < b ? 3 : 4;
static const vector<int, 2> vv = vector<int, 2>(a, b);

struct S
{
    bool member = a < b || c > (d);
    int lit = k<3>();
    int sum = k<a + b>();
    int calls = k<a>() + k<b>();
    vector<float, 3> v = vector<float, 3>(5.0);
};

int defaultArg(bool x = a < b || c > (d), int y = k<c>()) { return (x ? 10 : 0) + y; }

[numthreads(1, 1, 1)]
void computeMain()
{
    S s = S();
    outputBuffer[0] = global ? 1 : 0;
    outputBuffer[1] = tern;
    outputBuffer[2] = vv.y;
    outputBuffer[3] = s.member ? 1 : 0;
    outputBuffer[4] = s.lit;
    outputBuffer[5] = s.sum;
    outputBuffer[6] = s.calls;
    outputBuffer[7] = int(s.v.z);
    outputBuffer[8] = defaultArg();
}

// CHECK: type: int32_t
// CHECK-NEXT: 1
// CHECK-NEXT: 3
// CHECK-NEXT: 2
// CHECK-NEXT: 1
// CHECK-NEXT: 3
// CHECK-NEXT: 3
// CHECK-NEXT: 3
// CHECK-NEXT: 5
// CHECK-NEXT: 13
