// Like dyn-interface-nested-bwd.slang, but the outer dispatch is in a loop, so the backward pass
// checkpoints values of type `T.MinimalContext`, whose witness-table entries are still lazily
// translated placeholders.

//TEST:COMPARE_COMPUTE(targets=all, filecheck-buffer=CHECK):-output-using-type

//TEST_INPUT:ubuffer(data=[0 1], stride=4):name=inputBuffer
RWStructuredBuffer<uint> inputBuffer;

//TEST_INPUT:ubuffer(data=[0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

[anyValueSize(16)]
dyn interface IBsdf
{
    [Differentiable]
    float eval(float x);
};

struct Diffuse : IBsdf
{
    float k;

    [Differentiable, NoDiffThis]
    float eval(float x) { return k * x; }
};

struct Plastic : IBsdf
{
    float k;

    [Differentiable, NoDiffThis]
    float eval(float x) { return k * x * x; }
};

IBsdf makeBsdf(uint id)
{
    if (inputBuffer[id] == 0)
        return Diffuse(2.0f);
    return Plastic(3.0f);
}

[anyValueSize(16)]
dyn interface IIntegrator
{
    [Differentiable]
    float sample(float x);
};

struct Path : IIntegrator
{
    uint bsdfId;

    [Differentiable, NoDiffThis]
    float sample(float x)
    {
        IBsdf bsdf = no_diff makeBsdf(bsdfId);
        return bsdf.eval(x);
    }
};

struct Constant : IIntegrator
{
    [Differentiable, NoDiffThis]
    float sample(float x) { return x; }
};

IIntegrator makeIntegrator(uint id)
{
    if (inputBuffer[id] > 1)
        return Constant();
    return Path(id);
}

[Differentiable]
float render(IIntegrator integrator, float x)
{
    float result = 0.0f;
    for (uint i = 0; i < 2; i++)
        result += integrator.sample(x);
    return result;
}

[shader("compute"), numthreads(2, 1, 1)]
void computeMain(uint3 tid: SV_DispatchThreadID)
{
    IIntegrator integrator = makeIntegrator(tid.x);
    var dp = diffPair(1.0f, 0.0f);
    bwd_diff(render)(integrator, dp, 1.0f);
    outputBuffer[tid.x] = dp.d;
}

// Two iterations: Diffuse 2 * 2x -> 4, Plastic 2 * 3x^2 -> 12x = 12.
// CHECK: 4.0
// CHECK-NEXT: 12.0
