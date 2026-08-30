// The same shape as `dynamic-dispatch-nested-bwd.slang`, but with two implementations per
// interface, so the dispatch really is dynamic at run time and dispatch switches are emitted
// instead of devirtualizing. Runs on CPU compute rather than the interpreter because VM
// bytecode generation does not support the emitted dispatch code yet.

//TEST:COMPARE_COMPUTE(filecheck-buffer=CHECK):-output-using-type -cpu
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-vk -compute -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

public dyn interface IInner
{
    [Differentiable]
    float inner(float x);
}

public struct InnerA : IInner
{
    public float a;

    [Differentiable]
    [NoDiffThis]
    public float inner(float x) { return a * x * x; }
}

public struct InnerB : IInner
{
    public float a;

    [Differentiable]
    [NoDiffThis]
    public float inner(float x) { return a * x * x * x; }
}

public dyn interface IOuter
{
    [Differentiable]
    float outer(IInner i, float x);
}

public struct OuterA : IOuter
{
    public float k;

    [Differentiable]
    [NoDiffThis]
    public float outer(IInner i, float x) { return i.inner(x) * k; }
}

public struct OuterB : IOuter
{
    public float k;

    [Differentiable]
    [NoDiffThis]
    public float outer(IInner i, float x) { return i.inner(x) + k; }
}

[Differentiable]
float run(IOuter o, IInner i, float x)
{
    return o.outer(i, x);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    for (uint id = dispatchThreadID.x; id < 2; id++)
    {
        IInner i;
        IOuter o;
        if (id == 0)
        {
            i = InnerA(0.5f);
            o = OuterA(3.0f);
        }
        else
        {
            i = InnerB(1.5f);
            o = OuterB(1.0f);
        }

        var p = diffPair(2.0f);
        bwd_diff(run)(o, i, p, 1.0f);

        // id 0: run == 3 * (0.5 * x * x), so at x = 2 the value is 6 and d/dx is 6.
        // id 1: run == 1.5 * x * x * x + 1, so at x = 2 the value is 13 and d/dx is 18.
        outputBuffer[id * 2] = run(o, i, 2.0f);
        outputBuffer[id * 2 + 1] = p.d;
    }
    // CHECK: 6.0
    // CHECK-NEXT: 6.0
    // CHECK-NEXT: 13.0
    // CHECK-NEXT: 18.0
}
