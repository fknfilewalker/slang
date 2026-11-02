// Differentiate the same nested-dispatch method both through a dynamic call and through static
// calls. `Mesh.shade` dispatches on an `IBsdf` it creates, so its backward derivative is reached
// through the `IShape` witness table and, separately, through `outerMesh` and `outerGen<Mesh>`.
// Type-flow clones the backward derivative for the static calls, and the clone's call sites must
// keep their own callee signatures after the original function has been specialized.

//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-cpu -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-cuda -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

[anyValueSize(16)]
dyn interface IBsdf
{
    [Differentiable] float eval(float x);
}

struct Diffuse : IBsdf
{
    float k;
    [Differentiable, NoDiffThis] float eval(float x) { return k * x * x; } // 2kx
}

struct Plastic : IBsdf
{
    float k;
    [Differentiable, NoDiffThis] float eval(float x) { return k * x * x * x; } // 3kx^2
}

[anyValueSize(16)]
dyn interface IShape
{
    [Differentiable] float shade(float x);
}

struct Sphere : IShape
{
    uint b;
    [Differentiable, NoDiffThis] float shade(float x)
    {
        IBsdf bsdf = createDynamicObject<IBsdf>(b, float(b + 2)); // Diffuse k=2 or Plastic k=3
        return bsdf.eval(x * 2.0f);
    }
}

struct Mesh : IShape
{
    uint b;
    [Differentiable, NoDiffThis] float shade(float x)
    {
        IBsdf bsdf = createDynamicObject<IBsdf>(b, float(b + 2));
        return bsdf.eval(x) + x;
    }
}

[Differentiable] float outerDyn(IShape shape, float x) { return shape.shade(x); }
[Differentiable] float outerMesh(Mesh shape, float x) { return shape.shade(x); }
[Differentiable] float outerGen<T : IShape>(T shape, float x) { return shape.shade(x); }

//TEST_INPUT: type_conformance Diffuse:IBsdf = 0
//TEST_INPUT: type_conformance Plastic:IBsdf = 1
//TEST_INPUT: type_conformance Sphere:IShape = 0
//TEST_INPUT: type_conformance Mesh:IShape = 1

[numthreads(1, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    Mesh mesh;
    mesh.b = tid.x + 0;
    IShape shape = createDynamicObject<IShape>(tid.x + 0, tid.x + 0);

    // Sphere + Diffuse(k=2): d/dx 2(2x)^2 = 16x = 16
    var dp = diffPair(1.0f, 0.0f);
    bwd_diff(outerDyn)(shape, dp, 1.0f);
    outputBuffer[0] = dp.d;

    // Mesh + Diffuse(k=2): d/dx (2x^2 + x) = 4x + 1 = 5
    dp = diffPair(1.0f, 0.0f);
    bwd_diff(outerMesh)(mesh, dp, 1.0f);
    outputBuffer[1] = dp.d;

    dp = diffPair(1.0f, 0.0f);
    bwd_diff(outerGen<Mesh>)(mesh, dp, 1.0f);
    outputBuffer[2] = dp.d;
}

// CHECK: 16
// CHECK-NEXT: 5
// CHECK-NEXT: 5
