// An interface requirement takes `Optional<Ctx>`, with the associated type `Ctx` bound to a tuple
// holding an interface and to a float. The dispatcher unpacks the payload inside the optional when
// it has a value.

//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-cpu -compute -shaderobj -output-using-type
//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;
[anyValueSize(16)]
interface IA { float fa(float x); }
struct A0 : IA { float k; float fa(float x) { return k + x; } }
struct A1 : IA { float k; float fa(float x) { return k * x; } }
[anyValueSize(64)]
interface IShape
{
    associatedtype Ctx;
    Ctx make(uint a, float x);
    float useOpt(Optional<Ctx> c);
}
struct S0 : IShape
{
    typealias Ctx = Tuple<IA, float>;
    Ctx make(uint a, float x) { return makeTuple<IA, float>(createDynamicObject<IA>(a, x), 1.0); }
    float useOpt(Optional<Ctx> c) { if (c.hasValue) { IA p = c.value._0; return p.fa(2.0) + c.value._1; } return -1; }
}
struct S2 : IShape
{
    typealias Ctx = float;
    Ctx make(uint a, float x) { return x + a; }
    float useOpt(Optional<Ctx> c) { if (c.hasValue) return 2000 + c.value; return -2; }
}
float runOpt<S : IShape>(S s, uint a, float x, bool some)
{
    Optional<S.Ctx> c = none;
    if (some) c = s.make(a, x);
    return s.useOpt(c);
}
float run(IShape s, uint a, float x, bool some) { return runOpt(s, a, x, some); }
//TEST_INPUT: type_conformance A0:IA = 0
//TEST_INPUT: type_conformance A1:IA = 1
//TEST_INPUT: type_conformance S0:IShape = 0
//TEST_INPUT: type_conformance S2:IShape = 1
[numthreads(1, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    uint z = tid.x;
    outputBuffer[0] = run(createDynamicObject<IShape>(z + 0, 0u), z + 1, 2.0, true);  // A1(2).fa(2)+1 = 5
    outputBuffer[1] = run(createDynamicObject<IShape>(z + 0, 0u), z + 1, 2.0, false); // -1
    outputBuffer[2] = run(createDynamicObject<IShape>(z + 1, 0u), z + 1, 2.0, true);  // 2003
    outputBuffer[3] = run(createDynamicObject<IShape>(z + 1, 0u), z + 1, 2.0, false); // -2
}
// CHECK: 5
// CHECK-NEXT: -1
// CHECK-NEXT: 2003
// CHECK-NEXT: -2
