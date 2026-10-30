//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-cpu -output-using-type

// Inside a generic argument list, `>` ends the list. Inside brackets nested in it, it cannot, so
// `>`, `>=` and `>>` are operators there again.

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

int k<let N : int>() { return N; }

static const int a = 1, b = 2;

struct S
{
    int member = k<(a > b ? 1 : 2)>();
};

// An interface method with a default implementation is not a generic list.
interface IGet
{
    int get(bool x = b > a) { return x ? 9 : 8; }
}

struct Getter : IGet
{
}

[numthreads(1, 1, 1)]
void computeMain()
{
    outputBuffer[0] = k<(a > b)>();
    outputBuffer[1] = k<(a > b ? 1 : 2)>();
    outputBuffer[2] = k<(b >= a ? 3 : 4)>();
    outputBuffer[3] = k<(b >> 1)>();
    outputBuffer[4] = k<(8 >> a) + 1>();
    outputBuffer[5] = k<int(b > a)>();
    outputBuffer[6] = S().member;
    outputBuffer[7] = Getter().get();
}

// CHECK: type: int32_t
// CHECK-NEXT: 0
// CHECK-NEXT: 2
// CHECK-NEXT: 3
// CHECK-NEXT: 1
// CHECK-NEXT: 5
// CHECK-NEXT: 1
// CHECK-NEXT: 2
// CHECK-NEXT: 9
