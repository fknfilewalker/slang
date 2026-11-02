// An associated-type value is returned from and passed to helper generics typed `S.Ctx`, which
// are specialized with the set of `IShape` witness tables. The conformances bind `Ctx` to a
// concrete type, to an interface and to a tuple holding an interface, so the helpers' signatures
// and their callers must agree on one payload type per conformance.

//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-cpu -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
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

[anyValueSize(64)]
interface IShape
{
    associatedtype Ctx;
    Ctx make(uint a, float x);
    float use(Ctx c, float x);
}

struct S0 : IShape
{
    typealias Ctx = float;
    Ctx make(uint a, float x) { return a == 1 ? 2 * x : x + 2; }
    float use(Ctx c, float x) { return c + x; }
}

struct S1 : IShape
{
    typealias Ctx = IA;
    Ctx make(uint a, float x) { return createDynamicObject<IA>(a, x + 1); }
    float use(Ctx c, float x) { return 100 + c.fa(x); }
}

struct S2 : IShape
{
    typealias Ctx = Tuple<IA, float>;
    Ctx make(uint a, float x) { return makeTuple<IA, float>(createDynamicObject<IA>(a, x), x * 3); }
    float use(Ctx c, float x) { return 200 + c._0.fa(c._1); }
}

S.Ctx mk<S : IShape>(S s, uint a, float x) { return s.make(a, x); }
float pass<S : IShape>(S s, S.Ctx c, float x) { return s.use(c, x); }

float runGeneric<S : IShape>(S s, uint a, float x)
{
    S.Ctx c = mk(s, a, x);
    return pass(s, c, x);
}

float run(IShape s, uint a, float x) { return runGeneric(s, a, x); }

//TEST_INPUT: type_conformance A0:IA = 0
//TEST_INPUT: type_conformance A1:IA = 1
//TEST_INPUT: type_conformance S0:IShape = 0
//TEST_INPUT: type_conformance S1:IShape = 1
//TEST_INPUT: type_conformance S2:IShape = 2

[numthreads(1, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    uint z = tid.x;
    // S0: 2 * 2 + 2 = 6
    outputBuffer[0] = run(createDynamicObject<IShape>(z + 0, 0u), z + 1, 2.0);
    // S1 + A1(k=3): 100 + 3 * 2 = 106
    outputBuffer[1] = run(createDynamicObject<IShape>(z + 1, 0u), z + 1, 2.0);
    // S1 + A0(k=3): 100 + 3 + 2 = 105
    outputBuffer[2] = run(createDynamicObject<IShape>(z + 1, 0u), z + 0, 2.0);
    // S2 + A0(k=2), x tripled: 200 + 2 + 6 = 208
    outputBuffer[3] = run(createDynamicObject<IShape>(z + 2, 0u), z + 0, 2.0);
    // S2 + A1(k=2), x tripled: 200 + 2 * 6 = 212
    outputBuffer[4] = run(createDynamicObject<IShape>(z + 2, 0u), z + 1, 2.0);
}

// CHECK: 6
// CHECK-NEXT: 106
// CHECK-NEXT: 105
// CHECK-NEXT: 208
// CHECK-NEXT: 212
