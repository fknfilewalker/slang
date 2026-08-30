// Test reverse-mode differentiation of a dynamically dispatched interface *default method*.
// The default implementation is synthesized per conformance, so each synthesized method's
// intermediate context nests that conformance's own `eval` context (differently shaped across
// conformances, with a nested `IMaterial` dispatch inside `Area`'s). This combines default-method
// synthesis with the union narrowing exercised by dynamic-dispatch-bwd-diff-nested-existential.

//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-cpu -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

[anyValueSize(16)]
interface IMaterial
{
    [BackwardDifferentiable]
    float shade(float x);
}

struct Lambert : IMaterial
{
    float k;
    [BackwardDifferentiable]
    [NoDiffThis]
    float shade(float x) { return k * x; }
}

struct Shiny : IMaterial
{
    float k;
    float bias;
    [BackwardDifferentiable]
    [NoDiffThis]
    float shade(float x) { return k * x * x + bias; }
}

[anyValueSize(16)]
interface IEmitter
{
    [BackwardDifferentiable]
    float eval(IMaterial m, float x);

    // Default implementation: synthesized per conformance, so each conformance's context
    // nests its own eval context.
    [BackwardDifferentiable]
    float evalTwice(IMaterial m, float x)
    {
        float a = eval(m, x);
        float b = eval(m, x);
        return a + b;
    }
}

struct Area : IEmitter
{
    float scale;
    [BackwardDifferentiable]
    [NoDiffThis]
    float eval(IMaterial m, float x) { return m.shade(x) * scale; }
}

struct Point : IEmitter
{
    float intensity;
    [BackwardDifferentiable]
    [NoDiffThis]
    float eval(IMaterial m, float x) { return intensity * x * x; }
}

[BackwardDifferentiable]
float run(no_diff IEmitter e, no_diff IMaterial m, float x)
{
    float s = 0.0f;
    [MaxIters(2)]
    for (int i = 0; i < 2; ++i)
        s = s + e.evalTwice(m, x);
    return s;
}

IMaterial makeMaterial(uint id, float k)
{
    if (id == 0) { Lambert m = {k}; return m; }
    else { Shiny m = {k, 1.0f}; return m; }
}

IEmitter makeEmitter(uint id, float k)
{
    if (id == 0) { Area e = {k}; return e; }
    else { Point e = {k}; return e; }
}

//TEST_INPUT: type_conformance Lambert:IMaterial = 0
//TEST_INPUT: type_conformance Shiny:IMaterial = 1
//TEST_INPUT: type_conformance Area:IEmitter = 0
//TEST_INPUT: type_conformance Point:IEmitter = 1

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    // Area{2.0}, Lambert{0.5}: evalTwice = 2x, run = 2 iters -> 4x, d/dx = 4.
    var e = makeEmitter(dispatchThreadID.x, 2.0f);
    var m = makeMaterial(dispatchThreadID.x, 0.5f);
    var p = diffPair(3.0f);
    bwd_diff(run)(e, m, p, 1.0f);
    // CHECK: 12.0
    outputBuffer[0] = run(e, m, 3.0f);
    // CHECK: 4.0
    outputBuffer[1] = p.d;
}
