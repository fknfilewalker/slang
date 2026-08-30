// Regression test for a compiler hang in reverse-mode CFG normalization.
//
// `sample` opens the existential `rng` inside a loop body that also has an early `break`. Opening
// an existential to call a `[mutating]` method needs a temporary whose type is the *opened* type,
// so the loop body holds a local `var` of type `Ptr(extractExistentialType(...))` whose type
// instruction lives in that same block. Reverse-mode CFG normalization used to hoist every local
// out of such a region to the region header, which put the use of that block-local type before its
// definition; `legalizeDefUse` then re-ran forever trying to duplicate a hoistable type at its use
// site.

//TEST:SIMPLE(filecheck=CHECK): -target spirv -stage compute -entry computeMain
// CHECK: OpEntryPoint GLCompute

RWStructuredBuffer<float> outputBuffer;

interface IRandom
{
    [mutating]
    float next();
}

struct Rng : IRandom
{
    uint state;

    [mutating]
    float next()
    {
        state = state * 1664525u + 1013904223u;
        return float(state >> 8) * (1.0f / 16777216.0f);
    }
}

[BackwardDifferentiable]
float sample(float x, no_diff int count, inout IRandom rng)
{
    float acc = 0.0f;
    [MaxIters(8)]
    for (int i = 0; i < count; ++i)
    {
        float r = no_diff rng.next();
        acc = acc + x * r;
        if (acc > 10.0f)
            break;
    }
    return acc;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    Rng rng = {dispatchThreadID.x};
    IRandom r = rng;
    var x = diffPair(2.0f);
    bwd_diff(sample)(x, 4, r, 1.0f);
    outputBuffer[0] = x.d;
}
