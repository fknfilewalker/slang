// An associated type bound to `Optional<IBsdf>` is produced and consumed through a
// dynamically dispatched `IShape`, so the optional existential crosses the dispatch as the
// payload of the associated type.

//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-cpu -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

[anyValueSize(16)]
interface IBsdf
{
    float eval(float x);
}

struct Diffuse : IBsdf
{
    float k;
    float eval(float x) { return k * x; }
}

struct Plastic : IBsdf
{
    float k;
    float eval(float x) { return k * x * x; }
}

[anyValueSize(32)]
interface IShape
{
    associatedtype Ctx;
    Ctx make(float x);
    float use(Ctx c);
}

struct Sphere : IShape
{
    uint b;
    typealias Ctx = Optional<IBsdf>;
    Ctx make(float x)
    {
        IBsdf s = createDynamicObject<IBsdf>(b, float(b + 2)); // Diffuse(2) or Plastic(3)
        Optional<IBsdf> o = s; return o;
    }
    float use(Ctx c) { return c.value.eval(2.0); }
}

struct Mesh : IShape
{
    uint b;
    typealias Ctx = Optional<IBsdf>;
    Ctx make(float x)
    {
        IBsdf s = createDynamicObject<IBsdf>(b, float(b + 2));
        Optional<IBsdf> o = s; return o;
    }
    float use(Ctx c) { return c.value.eval(4.0) + 1; }
}

float runGeneric<T : IShape>(T shape, float x)
{
    let c = shape.make(x);
    return shape.use(c);
}

float run(IShape shape, float x)
{
    return runGeneric(shape, x);
}

//TEST_INPUT: type_conformance Diffuse:IBsdf = 0
//TEST_INPUT: type_conformance Plastic:IBsdf = 1
//TEST_INPUT: type_conformance Sphere:IShape = 0
//TEST_INPUT: type_conformance Mesh:IShape = 1

[numthreads(1, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    // Sphere + Diffuse(k=2): 2 * 2 = 4
    outputBuffer[0] = run(createDynamicObject<IShape>(tid.x + 0, tid.x + 0), 2.0);
    // Sphere + Plastic(k=3): 3 * 2 * 2 = 12
    outputBuffer[1] = run(createDynamicObject<IShape>(tid.x + 0, tid.x + 1), 2.0);
    // Mesh + Diffuse(k=2), x doubled: 2 * 4 + 1 = 9
    outputBuffer[2] = run(createDynamicObject<IShape>(tid.x + 1, tid.x + 0), 2.0);
    // Mesh + Plastic(k=3), x doubled: 3 * 4 * 4 + 1 = 49
    outputBuffer[3] = run(createDynamicObject<IShape>(tid.x + 1, tid.x + 1), 2.0);
}

// CHECK: 4
// CHECK-NEXT: 12
// CHECK-NEXT: 9
// CHECK-NEXT: 49
