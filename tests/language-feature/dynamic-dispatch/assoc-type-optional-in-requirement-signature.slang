// An interface requirement returns `Optional<Ctx>`, where the associated type `Ctx` is bound to a
// tuple holding an interface, to an array of an interface, and to a float. The value of the
// returned optional crosses the dynamic dispatch as a payload of the associated type.

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
    Optional<Ctx> makeOpt(uint a, float x);
    float use(Ctx c, float x);
}

struct S0 : IShape
{
    typealias Ctx = Tuple<IA, float>;
    Optional<Ctx> makeOpt(uint a, float x)
    {
        if (a > 1)
            return none;
        return makeTuple<IA, float>(createDynamicObject<IA>(a, x), 1.0);
    }
    float use(Ctx c, float x) { IA p = c._0; return p.fa(x) + c._1; }
}
struct S1 : IShape
{
    typealias Ctx = IA[1];
    Optional<Ctx> makeOpt(uint a, float x)
    {
        if (a > 1)
            return none;
        Ctx r = { createDynamicObject<IA>(a, x * 2) };
        return r;
    }
    float use(Ctx c, float x) { IA p = c[0]; return 1000 + p.fa(x); }
}
struct S2 : IShape
{
    typealias Ctx = float;
    Optional<Ctx> makeOpt(uint a, float x) { if (a > 1) return none; return x + a; }
    float use(Ctx c, float x) { return 2000 + c; }
}

float runGeneric<S : IShape>(S s, uint a, float x)
{
    let o = s.makeOpt(a, x);
    if (o.hasValue)
        return s.use(o.value, x);
    return -1;
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
    // S0 a=1 x=2: (A1(2), 1): 4 + 1 = 5
    outputBuffer[0] = run(createDynamicObject<IShape>(z + 0, 0u), z + 1, 2.0);
    // S0 a=0 x=1: (A0(1), 1): 2 + 1 = 3
    outputBuffer[1] = run(createDynamicObject<IShape>(z + 0, 0u), z + 0, 1.0);
    // S0 a=5: none
    outputBuffer[2] = run(createDynamicObject<IShape>(z + 0, 0u), z + 5, 1.0);
    // S1 a=1 x=2: {A1(4)}: 1000 + 8 = 1008
    outputBuffer[3] = run(createDynamicObject<IShape>(z + 1, 0u), z + 1, 2.0);
    // S2 a=1 x=2: 3: 2003
    outputBuffer[4] = run(createDynamicObject<IShape>(z + 2, 0u), z + 1, 2.0);
    // S2 a=5: none
    outputBuffer[5] = run(createDynamicObject<IShape>(z + 2, 0u), z + 5, 2.0);
}
// CHECK: 5
// CHECK-NEXT: 3
// CHECK-NEXT: -1
// CHECK-NEXT: 1008
// CHECK-NEXT: 2003
// CHECK-NEXT: -1
