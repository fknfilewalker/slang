// Differentiate through a dynamically dispatched `IEmitter` whose `Area` implementation obtains an
// `IShape` from a non-differentiable call and dispatches on it. The backward derivative of
// `Area.emit` saves the `IShape` existential in its minimal context, so the context type that
// `MinimalContext` selects for `Area` holds an interface. The witness-table entry for that type
// is a placeholder that autodiff translates lazily, and the payload type of the associated type
// must be computed from the translated tuple.

//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-cpu -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-dx12 -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-cuda -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

[anyValueSize(16)]
dyn interface IShape
{
    [Differentiable] float area(float x);
}

struct Sphere : IShape
{
    float r;
    [Differentiable, NoDiffThis] float area(float x) { return r * x * x; } // 2rx
}

struct Disk : IShape
{
    float r;
    [Differentiable, NoDiffThis] float area(float x) { return r * x * x * x; } // 3rx^2
}

[anyValueSize(32)]
dyn interface IEmitter
{
    [Differentiable] float emit(float x);
}

struct Point : IEmitter
{
    float p;
    [Differentiable, NoDiffThis] float emit(float x) { return p * x; } // p
}

IShape makeShape(uint id, float r)
{
    return createDynamicObject<IShape>(id, r);
}

struct Area : IEmitter
{
    uint shapeId;
    float r;
    [Differentiable, NoDiffThis]
    float emit(float x)
    {
        IShape shape = makeShape(shapeId, r);
        return shape.area(x) + shape.area(x * 2.0f);
    }
}

[Differentiable]
float accumulate(IEmitter emitter, float x, int n)
{
    float sum = 0.0f;
    [MaxIters(4)]
    for (int i = 0; i < n; i++)
        sum += emitter.emit(x);
    return sum;
}

//TEST_INPUT: type_conformance Sphere:IShape = 0
//TEST_INPUT: type_conformance Disk:IShape = 1
//TEST_INPUT: type_conformance Point:IEmitter = 0
//TEST_INPUT: type_conformance Area:IEmitter = 1

float bwd(IEmitter emitter, float x, int n)
{
    var dp = diffPair(x, 0.0f);
    bwd_diff(accumulate)(emitter, dp, n, 1.0f);
    return dp.d;
}

IEmitter makeArea(uint shapeId, float r)
{
    Area a;
    a.shapeId = shapeId;
    a.r = r;
    return a;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    // Point(p=3), 2 iterations: 2 * 3 = 6
    outputBuffer[0] = bwd(createDynamicObject<IEmitter>(tid.x + 0, 3.0f), 1.0f, 2);
    // Area + Sphere(r=2): d/dx (2x^2 + 8x^2) = 20x; x = 1, 2 iterations: 40
    outputBuffer[1] = bwd(makeArea(tid.x + 0, 2.0f), 1.0f, 2);
    // Area + Disk(r=1): d/dx (x^3 + 8x^3) = 27x^2; x = 1, 3 iterations: 81
    outputBuffer[2] = bwd(makeArea(tid.x + 1, 1.0f), 1.0f, 3);
    // Area + Disk(r=2): d/dx (2x^3 + 16x^3) = 54x^2; x = 1, 1 iteration: 54
    outputBuffer[3] = bwd(makeArea(tid.x + 1, 2.0f), 1.0f, 1);
}

// CHECK: 6
// CHECK-NEXT: 40
// CHECK-NEXT: 81
// CHECK-NEXT: 54
