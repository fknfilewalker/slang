// An associated type is bound to `Tuple<Optional<IB>, float>`, where no type conforms to `IB`.
// The optional is always `none`, which is a valid value, so the payload type of the associated
// type holds a tagged union over just the `none` witness rather than being rejected.

//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-cpu -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

[anyValueSize(16)]
interface IA
{
    float fa(float x);
}

struct A0 : IA
{
    float k;
    float fa(float x) { return k + x; }
}

struct A1 : IA
{
    float k;
    float fa(float x) { return k * x; }
}

[anyValueSize(16)]
interface IB
{
    float fb(float x);
}

[anyValueSize(64)]
interface IShape
{
    associatedtype Ctx;
    Ctx make(uint a, float x);
    float use(Ctx c, float x);
}

struct S0 : IShape
{
    typealias Ctx = Tuple<IA, float>;
    Ctx make(uint a, float x) { return makeTuple<IA, float>(createDynamicObject<IA>(a, x), x); }
    float use(Ctx c, float x) { return c._0.fa(x) + c._1; }
}

struct S1 : IShape
{
    typealias Ctx = Tuple<Optional<IB>, float>;
    Ctx make(uint a, float x) { return makeTuple<Optional<IB>, float>(none, x); }
    float use(Ctx c, float x) { return c._0.hasValue ? -1 : 100 + c._1; }
}

float runGeneric<S : IShape>(S s, uint a, float x)
{
    let c = s.make(a, x);
    return s.use(c, x);
}

float run(IShape s, uint a, float x) { return runGeneric(s, a, x); }

//TEST_INPUT: type_conformance A0:IA = 0
//TEST_INPUT: type_conformance A1:IA = 1
//TEST_INPUT: type_conformance S0:IShape = 0
//TEST_INPUT: type_conformance S1:IShape = 1

[numthreads(1, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    // S0 + A1(k=2): 2 * 2 + 2 = 6
    outputBuffer[0] = run(createDynamicObject<IShape>(tid.x + 0, 0u), tid.x + 1, 2.0);
    // S1: the optional is none, so 100 + 2 = 102
    outputBuffer[1] = run(createDynamicObject<IShape>(tid.x + 1, 0u), tid.x + 1, 2.0);
}

// CHECK: 6
// CHECK-NEXT: 102
