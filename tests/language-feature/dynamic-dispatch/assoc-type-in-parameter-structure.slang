// An interface requirement takes the associated type `Ctx` inside a tuple, an array and an
// optional parameter, with `Ctx` bound to a tuple holding an interface, to an interface, and to a
// float. The dispatcher receives each such parameter with an untagged-union payload in place of
// `Ctx`, and unpacks it into each implementation's own parameter type.

//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-cpu -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0], stride=4):out,name=outputBuffer
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
    float useTuple(Tuple<Ctx, float> c);
    float useArray(Ctx c[2]);
}

struct S0 : IShape
{
    typealias Ctx = Tuple<IA, float>;
    Ctx make(uint a, float x) { return makeTuple<IA, float>(createDynamicObject<IA>(a, x), 1.0); }
    float useTuple(Tuple<Ctx, float> c) { IA p = c._0._0; return p.fa(c._1) + c._0._1; }
    float useArray(Ctx c[2]) { IA p = c[0]._0; IA q = c[1]._0; return p.fa(1.0) * 10 + q.fa(1.0); }
}
struct S1 : IShape
{
    typealias Ctx = IA;
    Ctx make(uint a, float x) { return createDynamicObject<IA>(a, x * 2); }
    float useTuple(Tuple<Ctx, float> c) { return 1000 + c._0.fa(c._1); }
    float useArray(Ctx c[2]) { return 1000 + c[0].fa(1.0) * 10 + c[1].fa(1.0); }
}
struct S2 : IShape
{
    typealias Ctx = float;
    Ctx make(uint a, float x) { return x + a; }
    float useTuple(Tuple<Ctx, float> c) { return 2000 + c._0 * c._1; }
    float useArray(Ctx c[2]) { return 2000 + c[0] * 10 + c[1]; }
}

float runTuple<S : IShape>(S s, uint a, float x)
{
    let c = s.make(a, x);
    return s.useTuple(makeTuple(c, 3.0f));
}
float runArray<S : IShape>(S s, uint a, float x)
{
    S.Ctx c[2] = { s.make(a, x), s.make(1 - a, x + 1) };
    return s.useArray(c);
}
float runT(IShape s, uint a, float x) { return runTuple(s, a, x); }
float runA(IShape s, uint a, float x) { return runArray(s, a, x); }

//TEST_INPUT: type_conformance A0:IA = 0
//TEST_INPUT: type_conformance A1:IA = 1
//TEST_INPUT: type_conformance S0:IShape = 0
//TEST_INPUT: type_conformance S1:IShape = 1
//TEST_INPUT: type_conformance S2:IShape = 2

[numthreads(1, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    uint z = tid.x;
    // S0 a=1 x=2: A1(2).fa(3) + 1 = 7
    outputBuffer[0] = runT(createDynamicObject<IShape>(z + 0, 0u), z + 1, 2.0);
    // S1 a=0 x=2: A0(4).fa(3) = 7 -> 1007
    outputBuffer[1] = runT(createDynamicObject<IShape>(z + 1, 0u), z + 0, 2.0);
    // S2 a=1 x=2: 2000 + 3 * 3 = 2009
    outputBuffer[2] = runT(createDynamicObject<IShape>(z + 2, 0u), z + 1, 2.0);
    // S0 a=0 x=2: {A0(2), A1(3)}: 3 * 10 + 3 = 33
    outputBuffer[3] = runA(createDynamicObject<IShape>(z + 0, 0u), z + 0, 2.0);
    // S1 a=1 x=2: {A1(4), A0(6)}: 1000 + 4 * 10 + 7 = 1047
    outputBuffer[4] = runA(createDynamicObject<IShape>(z + 1, 0u), z + 1, 2.0);
    // S2 a=1 x=2: {3, 3}: 2000 + 30 + 3 = 2033
    outputBuffer[5] = runA(createDynamicObject<IShape>(z + 2, 0u), z + 1, 2.0);
}
// CHECK: 7
// CHECK-NEXT: 1007
// CHECK-NEXT: 2009
// CHECK-NEXT: 33
// CHECK-NEXT: 1047
// CHECK-NEXT: 2033
