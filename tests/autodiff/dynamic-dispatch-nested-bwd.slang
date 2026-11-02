// Differentiate through a dynamically dispatched method that dispatches again on an
// interface it creates. The inner receiver is saved in the minimal context, which crosses
// the outer dispatch through the `MinimalContext`/`BwdCallable` associated types.

//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-cpu -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-dx12 -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-cuda -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0], stride=4):out,name=outputBuffer
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
    [Differentiable] float shade(float x);
};

struct Sphere : IShape
{
    uint b;
    [Differentiable, NoDiffThis] float shade(float x)
    {
        IBsdf bsdf = createDynamicObject<IBsdf>(b, float(b + 2)); // Diffuse k=2 or Plastic k=3
        return bsdf.eval(x * 2.0f);
    }
};

struct Mesh : IShape
{
    uint b;
    [Differentiable, NoDiffThis] float shade(float x)
    {
        IBsdf bsdf = createDynamicObject<IBsdf>(b, float(b + 2));
        return bsdf.eval(x) + x;
    }
};

[Differentiable]
float outer(IShape shape, float x)
{
    return shape.shade(x);
}

//TEST_INPUT: type_conformance Diffuse:IBsdf = 0
//TEST_INPUT: type_conformance Plastic:IBsdf = 1
//TEST_INPUT: type_conformance Sphere:IShape = 0
//TEST_INPUT: type_conformance Mesh:IShape = 1

float bwd(IShape shape, float x)
{
    var dp = diffPair(x, 0.0f);
    __bwd_diff(outer)(shape, dp, 1.0f);
    return dp.d;
}

float fwd(IShape shape, float x)
{
    return __fwd_diff(outer)(shape, diffPair(x, 1.0f)).d;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    // Sphere + Diffuse(k=2): 2 (2x)^2 = 8x^2 -> 16x; x = 1.5 -> 24
    IShape s0 = createDynamicObject<IShape>(tid.x + 0, tid.x + 0);
    outputBuffer[0] = bwd(s0, 1.5f);
    outputBuffer[1] = fwd(s0, 1.5f);
    // Sphere + Plastic(k=3): 3 (2x)^3 = 24x^3 -> 72x^2; x = 0.5 -> 18
    IShape s1 = createDynamicObject<IShape>(tid.x + 0, tid.x + 1);
    outputBuffer[2] = bwd(s1, 0.5f);
    outputBuffer[3] = fwd(s1, 0.5f);
    // Mesh + Diffuse(k=2): 2x^2 + x -> 4x + 1; x = 2 -> 9
    IShape s2 = createDynamicObject<IShape>(tid.x + 1, tid.x + 0);
    outputBuffer[4] = bwd(s2, 2.0f);
    outputBuffer[5] = fwd(s2, 2.0f);
    // Mesh + Plastic(k=3): 3x^3 + x -> 9x^2 + 1; x = 1 -> 10
    IShape s3 = createDynamicObject<IShape>(tid.x + 1, tid.x + 1);
    outputBuffer[6] = bwd(s3, 1.0f);
    outputBuffer[7] = fwd(s3, 1.0f);
}

// CHECK: 24
// CHECK-NEXT: 24
// CHECK-NEXT: 18
// CHECK-NEXT: 18
// CHECK-NEXT: 9
// CHECK-NEXT: 9
// CHECK-NEXT: 10
// CHECK-NEXT: 10
