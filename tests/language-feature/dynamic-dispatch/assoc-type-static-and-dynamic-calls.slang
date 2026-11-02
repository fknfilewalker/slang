// The producer and consumer of an associated type are called both statically, through
// `rg<S0>` and `rg<S1>`, and dynamically, through `rg` specialized with the set of `IShape`
// witness tables. `S1` binds `Ctx` to an interface, so an earlier type-flow run lowers the result
// of `S1.make` to a tagged union before a later run sees the dynamic call to it again; that
// lowered result must still cross the dispatch as the payload of the associated type.

//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-cpu -compute -shaderobj -output-using-type
//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;
[anyValueSize(16)] interface IA { float fa(float x); }
struct A0 : IA { float k; float fa(float x) { return k + x; } }
struct A1 : IA { float k; float fa(float x) { return k * x; } }
struct A2 : IA { float k; float2 big; float fa(float x) { return k - x + big.y; } }
[anyValueSize(64)]
interface IShape { associatedtype Ctx; Ctx make(uint a, float x); float use(Ctx c, float x); }
struct S0 : IShape
{
    typealias Ctx = Tuple<IA, float>;
    Ctx make(uint a, float x)
    {
        if (a == 9) { A0 c; c.k = x; return makeTuple<IA, float>(c, 1.0); }   // statically A0
        return makeTuple<IA, float>(createDynamicObject<IA>(a, x), 2.0);
    }
    float use(Ctx c, float x) { IA p = c._0; return p.fa(x) + c._1; }
}
struct S1 : IShape
{
    typealias Ctx = IA;
    Ctx make(uint a, float x) { A1 r; r.k = x + a; return r; }
    float use(Ctx c, float x) { return 100 + c.fa(x); }
}
float rg<S : IShape>(S s, uint a, float x) { let c = s.make(a, x); return s.use(c, x); }
float run(IShape s, uint a, float x) { return rg(s, a, x); }
//TEST_INPUT: type_conformance A0:IA = 0
//TEST_INPUT: type_conformance A1:IA = 1
//TEST_INPUT: type_conformance A2:IA = 2
//TEST_INPUT: type_conformance S0:IShape = 0
//TEST_INPUT: type_conformance S1:IShape = 1
[numthreads(1, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    uint z = tid.x;
    S0 s0; S1 s1;
    outputBuffer[0] = rg(s0, 9, 2.0);                                         // A0(2): 4 + 1 = 5
    outputBuffer[1] = rg(s0, z + 1, 2.0);                                     // A1(2): 4 + 2 = 6
    outputBuffer[2] = run(createDynamicObject<IShape>(z + 0, 0u), 9, 3.0);    // A0(3): 6 + 1 = 7
    outputBuffer[3] = run(createDynamicObject<IShape>(z + 0, 0u), z + 1, 3.0);// A1(3): 9 + 2 = 11
    outputBuffer[4] = rg(s1, 1, 2.0);                                         // 100 + 3*2 = 106
    outputBuffer[5] = run(createDynamicObject<IShape>(z + 1, 0u), 1, 2.0) + s0.use(s0.make(z, 1.0), 1.0); // 106 + (2 + 2) = 110
}
// CHECK: 5
// CHECK-NEXT: 6
// CHECK-NEXT: 7
// CHECK-NEXT: 11
// CHECK-NEXT: 106
// CHECK-NEXT: 110
