// An interface requirement passes the associated type `Ctx` as an `out` parameter, as an `inout`
// parameter, and inside an `out` tuple, with `Ctx` bound to a tuple holding an interface, to an
// interface, and to a float. The dispatcher unpacks each value into the implementation's
// parameter type and packs it back into the payload after the call.

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
    void make(uint a, float x, out Ctx c);
    void makeInTuple(uint a, float x, out Tuple<Ctx, float> c);
    void flip(inout Ctx c);
    float use(Ctx c);
}

struct S0 : IShape
{
    typealias Ctx = Tuple<IA, float>;
    void make(uint a, float x, out Ctx c) { c = makeTuple<IA, float>(createDynamicObject<IA>(a, x), 1.0); }
    void makeInTuple(uint a, float x, out Tuple<Ctx, float> c) { Ctx inner; make(a, x, inner); c = makeTuple(inner, 5.0f); }
    void flip(inout Ctx c) { c = makeTuple<IA, float>(createDynamicObject<IA>(1, c._1 + 1), c._1 + 1); }
    float use(Ctx c) { IA p = c._0; return p.fa(2.0) + c._1; }
}
struct S1 : IShape
{
    typealias Ctx = IA;
    void make(uint a, float x, out Ctx c) { c = createDynamicObject<IA>(a, x * 2); }
    void makeInTuple(uint a, float x, out Tuple<Ctx, float> c) { c = makeTuple<IA, float>(createDynamicObject<IA>(a, x * 2), 5.0f); }
    void flip(inout Ctx c) { c = createDynamicObject<IA>(0, c.fa(1.0)); }
    float use(Ctx c) { return 1000 + c.fa(2.0); }
}
struct S2 : IShape
{
    typealias Ctx = float;
    void make(uint a, float x, out Ctx c) { c = x + a; }
    void makeInTuple(uint a, float x, out Tuple<Ctx, float> c) { c = makeTuple(x + a, 5.0f); }
    void flip(inout Ctx c) { c = -c; }
    float use(Ctx c) { return 2000 + c; }
}

float runOut<S : IShape>(S s, uint a, float x)
{
    S.Ctx c;
    s.make(a, x, c);
    s.flip(c);
    return s.use(c);
}
float runInTuple<S : IShape>(S s, uint a, float x)
{
    Tuple<S.Ctx, float> c;
    s.makeInTuple(a, x, c);
    return s.use(c._0) + c._1 * 100;
}
float runO(IShape s, uint a, float x) { return runOut(s, a, x); }
float runT(IShape s, uint a, float x) { return runInTuple(s, a, x); }

//TEST_INPUT: type_conformance A0:IA = 0
//TEST_INPUT: type_conformance A1:IA = 1
//TEST_INPUT: type_conformance S0:IShape = 0
//TEST_INPUT: type_conformance S1:IShape = 1
//TEST_INPUT: type_conformance S2:IShape = 2

[numthreads(1, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    uint z = tid.x;
    // S0: (A0(2), 1) flipped to (A1(2), 2): 2 * 2 + 2 = 6
    outputBuffer[0] = runO(createDynamicObject<IShape>(z + 0, 0u), z + 0, 2.0);
    // S1: A1(4) flipped to A0(4): 1000 + 4 + 2 = 1006
    outputBuffer[1] = runO(createDynamicObject<IShape>(z + 1, 0u), z + 1, 2.0);
    // S2: 3 flipped to -3: 1997
    outputBuffer[2] = runO(createDynamicObject<IShape>(z + 2, 0u), z + 1, 2.0);
    // S0: ((A1(2), 1), 5): 2 * 2 + 1 + 500 = 505
    outputBuffer[3] = runT(createDynamicObject<IShape>(z + 0, 0u), z + 1, 2.0);
    // S1: (A0(4), 5): 1000 + 6 + 500 = 1506
    outputBuffer[4] = runT(createDynamicObject<IShape>(z + 1, 0u), z + 0, 2.0);
    // S2: (3, 5): 2003 + 500 = 2503
    outputBuffer[5] = runT(createDynamicObject<IShape>(z + 2, 0u), z + 1, 2.0);
}
// CHECK: 6
// CHECK-NEXT: 1006
// CHECK-NEXT: 1997
// CHECK-NEXT: 505
// CHECK-NEXT: 1506
// CHECK-NEXT: 2503
