// Reverse-mode differentiation through a dynamically dispatched method that dispatches
// again on an interface parameter. The parameter is saved in the parameter context, which
// crosses the outer dispatch inside the full context (`BwdCallable`).

//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-cpu -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-dx12 -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-cuda -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

[anyValueSize(16)]
dyn interface IBsdf
{
    [Differentiable] float eval(float x);
};

struct Diffuse : IBsdf
{
    float k;
    [Differentiable, NoDiffThis] float eval(float x) { return k * x * x; } // 2kx
};

struct Plastic : IBsdf
{
    float k;
    [Differentiable, NoDiffThis] float eval(float x) { return k * x * x * x; } // 3kx^2
};

[anyValueSize(16)]
dyn interface IShape
{
    [Differentiable] float shade(IBsdf bsdf, float x);
};

struct Sphere : IShape
{
    uint pad;
    [Differentiable, NoDiffThis] float shade(IBsdf bsdf, float x) { return bsdf.eval(x * 2.0f); }
};

struct Mesh : IShape
{
    uint pad;
    [Differentiable, NoDiffThis] float shade(IBsdf bsdf, float x) { return bsdf.eval(x) + x; }
};

[Differentiable]
float outer(IShape shape, IBsdf bsdf, float x)
{
    return shape.shade(bsdf, x);
}

//TEST_INPUT: type_conformance Diffuse:IBsdf = 0
//TEST_INPUT: type_conformance Plastic:IBsdf = 1
//TEST_INPUT: type_conformance Sphere:IShape = 0
//TEST_INPUT: type_conformance Mesh:IShape = 1

float bwd(IShape shape, IBsdf bsdf, float x)
{
    var dp = diffPair(x, 0.0f);
    __bwd_diff(outer)(shape, bsdf, dp, 1.0f);
    return dp.d;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    IBsdf diffuse = createDynamicObject<IBsdf>(tid.x + 0, 2.0f);
    IBsdf plastic = createDynamicObject<IBsdf>(tid.x + 1, 3.0f);
    IShape sphere = createDynamicObject<IShape>(tid.x + 0, 0u);
    IShape mesh = createDynamicObject<IShape>(tid.x + 1, 0u);

    // Sphere + Diffuse(k=2): 2 (2x)^2 = 8x^2 -> 16x; x = 1.5 -> 24
    outputBuffer[0] = bwd(sphere, diffuse, 1.5f);
    // Sphere + Plastic(k=3): 3 (2x)^3 = 24x^3 -> 72x^2; x = 0.5 -> 18
    outputBuffer[1] = bwd(sphere, plastic, 0.5f);
    // Mesh + Diffuse(k=2): 2x^2 + x -> 4x + 1; x = 2 -> 9
    outputBuffer[2] = bwd(mesh, diffuse, 2.0f);
    // Mesh + Plastic(k=3): 3x^3 + x -> 9x^2 + 1; x = 1 -> 10
    outputBuffer[3] = bwd(mesh, plastic, 1.0f);
}

// CHECK: 24
// CHECK-NEXT: 18
// CHECK-NEXT: 9
// CHECK-NEXT: 10
