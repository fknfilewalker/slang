// An interface requirement returns `Ctx[2]`, where the associated type `Ctx` is bound to a tuple
// holding an interface, to an interface, and to a float. Each element of the returned array
// crosses the dynamic dispatch as a payload of the associated type.

//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-cpu -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

[anyValueSize(16)]
interface IA { float fa(float x); }
struct A0 : IA { float k; float fa(float x) { return k + x; } }
struct A1 : IA { float k; float fa(float x) { return k * x; } }

[anyValueSize(64)]
interface IShape
{
    associatedtype Ctx;
    Ctx[2] makeArr(uint a, float x);
    float use(Ctx c, float x);
}

struct S0 : IShape
{
    typealias Ctx = Tuple<IA, float>;
    Ctx make(uint a, float x) { return makeTuple<IA, float>(createDynamicObject<IA>(a, x), 1.0); }
    Ctx[2] makeArr(uint a, float x) { Ctx r[2] = { make(a, x), make(1 - a, x) }; return r; }
    float use(Ctx c, float x) { IA p = c._0; return p.fa(x) + c._1; }
}
struct S1 : IShape
{
    typealias Ctx = IA;
    Ctx make(uint a, float x) { return createDynamicObject<IA>(a, x * 2); }
    Ctx[2] makeArr(uint a, float x) { Ctx r[2] = { make(a, x), make(1 - a, x) }; return r; }
    float use(Ctx c, float x) { return 1000 + c.fa(x); }
}
struct S2 : IShape
{
    typealias Ctx = float;
    Ctx[2] makeArr(uint a, float x) { Ctx r[2] = { x + a, x - a }; return r; }
    float use(Ctx c, float x) { return 2000 + c; }
}

float runGeneric<S : IShape>(S s, uint a, float x)
{
    let arr = s.makeArr(a, x);
    return s.use(arr[0], x) + s.use(arr[1], x);
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
    // S0 a=1 x=2: arr = {(A1(2),1), (A0(2),1)}: (4 + 1) + (4 + 1) = 10
    outputBuffer[0] = run(createDynamicObject<IShape>(z + 0, 0u), z + 1, 2.0);
    // S0 a=0 x=1: arr = {(A0(1),1), (A1(1),1)}: (2 + 1) + (1 + 1) = 5
    outputBuffer[1] = run(createDynamicObject<IShape>(z + 0, 0u), z + 0, 1.0);
    // S1 a=0 x=1: arr = {A0(2), A1(2)}: (1000 + 3) + (1000 + 2) = 2005
    outputBuffer[2] = run(createDynamicObject<IShape>(z + 1, 0u), z + 0, 1.0);
    // S1 a=1 x=2: arr = {A1(4), A0(4)}: (1000 + 8) + (1000 + 6) = 2014
    outputBuffer[3] = run(createDynamicObject<IShape>(z + 1, 0u), z + 1, 2.0);
    // S2 a=1 x=2: arr = {3, 1}: 2003 + 2001 = 4004
    outputBuffer[4] = run(createDynamicObject<IShape>(z + 2, 0u), z + 1, 2.0);
}
// CHECK: 10
// CHECK-NEXT: 5
// CHECK-NEXT: 2005
// CHECK-NEXT: 2014
// CHECK-NEXT: 4004
