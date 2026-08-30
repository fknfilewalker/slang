// Test reverse-mode differentiation of a dynamically dispatched method whose conformances have
// differently-shaped intermediate contexts, one of which embeds a nested existential dispatch.
//
// The reverse-mode split stores each loop iteration's intermediate context of the dispatched
// `eval` call. Because the two conformances' propagate functions take differently-shaped contexts
// (`Area`'s captures the nested `IMaterial` dispatch, `Point`'s does not), the stored slot is an
// untagged union over both context types, while each dispatch branch passes the stored value to a
// parameter declared with that one conformance's own context type. Type-flow specialization must
// narrow the union info to the compatible member at such a parameter (and refine the member's
// nested existential leaves to their conformance sets) rather than flat-unioning the box into the
// parameter; getting this wrong widens the parameter into an opaque any-value that the propagate
// body still indexes as a tuple.

//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-cpu -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
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
}

// A conformance whose differentiated method dispatches through the nested existential, so its
// intermediate context captures the material's context and dispatch tag.
struct Area : IEmitter
{
    float scale;
    [BackwardDifferentiable]
    [NoDiffThis]
    float eval(IMaterial m, float x) { return m.shade(x) * scale; }
}

// A conformance with a plain context of a different shape that ignores the material.
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
        s = s + e.eval(m, x);
    return s;
}

IMaterial makeMaterial(uint id, float k)
{
    if (id == 0)
    {
        Lambert m = {k};
        return m;
    }
    else
    {
        Shiny m = {k, 1.0f};
        return m;
    }
}

IEmitter makeEmitter(uint id, float k)
{
    if (id == 0)
    {
        Area e = {k};
        return e;
    }
    else
    {
        Point e = {k};
        return e;
    }
}

//TEST_INPUT: type_conformance Lambert:IMaterial = 0
//TEST_INPUT: type_conformance Shiny:IMaterial = 1
//TEST_INPUT: type_conformance Area:IEmitter = 0
//TEST_INPUT: type_conformance Point:IEmitter = 1

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    // Area{2.0} with Lambert{0.5}: eval = 0.5*x*2 = x, run = 2 iterations -> 2x, d/dx = 2.
    var e = makeEmitter(dispatchThreadID.x, 2.0f);
    var m = makeMaterial(dispatchThreadID.x, 0.5f);
    var p = diffPair(3.0f);
    bwd_diff(run)(e, m, p, 1.0f);
    // CHECK: 6.0
    outputBuffer[0] = run(e, m, 3.0f);
    // CHECK: 2.0
    outputBuffer[1] = p.d;

    // Point{1.5} (ignores the material): eval = 1.5*x^2, run -> 3x^2, d/dx = 6x = 18 at x = 3.
    e = makeEmitter(dispatchThreadID.x + 1, 1.5f);
    m = makeMaterial(dispatchThreadID.x + 1, 0.5f);
    p = diffPair(3.0f);
    bwd_diff(run)(e, m, p, 1.0f);
    // CHECK: 27.0
    outputBuffer[2] = run(e, m, 3.0f);
    // CHECK: 18.0
    outputBuffer[3] = p.d;
}
