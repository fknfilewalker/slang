// Regression test for reverse-mode normalization of a local existential whose opened type is
// referenced outside the block that defines it. Def-use legalization must make the transitive
// operands of the opened pointer type available, and subsequent type-flow specialization must
// converge after lowering the temporary existential storage.

//TEST:SIMPLE(filecheck=CHECK): -target spirv -stage compute -entry computeMain
// CHECK: OpEntryPoint GLCompute

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

interface IOp
{
    [BackwardDifferentiable]
    float eval(float x);
}

struct Square : IOp
{
    [BackwardDifferentiable]
    float eval(float x)
    {
        return x * x;
    }
}

[BackwardDifferentiable]
float evaluate(no_diff IOp op, float x)
{
    IOp selected = op;
    for (int i = 0; i < 2; ++i)
    {
        if (i == 1)
            return selected.eval(x);
    }
    return 0.0f;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    IOp op = Square();
    var x = diffPair(2.0f);
    bwd_diff(evaluate)(op, x, 1.0f);
    outputBuffer[0] = x.d;
}
