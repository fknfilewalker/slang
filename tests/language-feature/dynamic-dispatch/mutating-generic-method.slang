//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-cpu -output-using-type

// A `[mutating]` generic method called on an `inout` interface value must update
// that value, like a non-generic method does. It used to update a copy instead:
// the parser checks `r.next` early to decide whether `<` starts generic arguments,
// and dropped the binding that makes the temporary for `r` refer to `r` itself.

//TEST_INPUT:ubuffer(data=[0 1], stride=4):name=inputs
RWStructuredBuffer<uint> inputs;

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<uint> outputBuffer;

dyn interface ICounter
{
    [mutating] uint next();
    [mutating] uint next<let N : int>();
};

struct A : ICounter
{
    uint s;
    [mutating] uint next() { s += 1; return s; }
    [mutating] uint next<let N : int>() { s += N; return s; }
};

struct B : ICounter
{
    uint s;
    [mutating] uint next() { s += 10; return s; }
    [mutating] uint next<let N : int>() { s += 10 * N; return s; }
};

ICounter make(uint k)
{
    if (k == 0)
        return A(0);
    return B(0);
}

uint2 generic2(inout ICounter r)
{
    let a = r.next<2>();
    let b = r.next<2>();
    return uint2(a, b);
}

uint2 nonGeneric(inout ICounter r)
{
    let a = r.next();
    let b = r.next();
    return uint2(a, b);
}

[numthreads(1, 1, 1)]
void computeMain()
{
    for (uint i = 0; i < 2; i++)
    {
        var r = make(inputs[i]);
        let g = generic2(r);
        let n = nonGeneric(r);
        outputBuffer[i * 4 + 0] = g.x;
        outputBuffer[i * 4 + 1] = g.y;
        outputBuffer[i * 4 + 2] = n.x;
        outputBuffer[i * 4 + 3] = n.y;
    }
}

// CHECK: type: uint32_t
// CHECK-NEXT: 2
// CHECK-NEXT: 4
// CHECK-NEXT: 5
// CHECK-NEXT: 6
// CHECK-NEXT: 20
// CHECK-NEXT: 40
// CHECK-NEXT: 50
// CHECK-NEXT: 60
