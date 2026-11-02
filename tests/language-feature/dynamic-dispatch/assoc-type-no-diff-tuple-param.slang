// A differentiable interface requirement takes the associated type `Ctx` inside a `no_diff`
// tuple parameter, with `Ctx` bound to a tuple holding an interface, to an interface, and to a
// float. The attribute wraps the tuple, and the payload of `Ctx` sits in its first element.

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
    Ctx make(uint a, float x);
    [Differentiable] float use(no_diff Tuple<Ctx, float> c, float x);
}

struct S0 : IShape
{
    typealias Ctx = Tuple<IA, float>;
    Ctx make(uint a, float x) { return makeTuple<IA, float>(createDynamicObject<IA>(a, x), 1.0); }
    [Differentiable] float use(no_diff Tuple<Ctx, float> c, float x) { IA p = c._0._0; return no_diff(p.fa(detach(x))) + c._0._1 * x + c._1; }
}
struct S1 : IShape
{
    typealias Ctx = IA;
    Ctx make(uint a, float x) { return createDynamicObject<IA>(a, x * 2); }
    [Differentiable] float use(no_diff Tuple<Ctx, float> c, float x) { return 1000 + no_diff(c._0.fa(detach(x))) + x + c._1; }
}
struct S2 : IShape
{
    typealias Ctx = float;
    Ctx make(uint a, float x) { return x + a; }
    [Differentiable] float use(no_diff Tuple<Ctx, float> c, float x) { return 2000 + c._0 + x + c._1; }
}

float runGeneric<S : IShape>(S s, uint a, float x)
{
    let c = s.make(a, x);
    return s.use(makeTuple(c, 0.5f), x);
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
    // S0 a=1 x=2: A1(2).fa(2)=4, + 1*2 = 6
    outputBuffer[0] = run(createDynamicObject<IShape>(z + 0, 0u), z + 1, 2.0);
    // S0 a=0 x=1: A0(1).fa(1)=2, + 1 = 3
    outputBuffer[1] = run(createDynamicObject<IShape>(z + 0, 0u), z + 0, 1.0);
    // S1 a=0 x=1: A0(2).fa(1)=3 -> 1000+3+1 = 1004.5
    outputBuffer[2] = run(createDynamicObject<IShape>(z + 1, 0u), z + 0, 1.0);
    // S1 a=1 x=2: A1(4).fa(2)=8 -> 1000+8+2 = 1010.5
    outputBuffer[3] = run(createDynamicObject<IShape>(z + 1, 0u), z + 1, 2.0);
    // S2 a=1 x=2: 2000+3+2 = 2005.5
    outputBuffer[4] = run(createDynamicObject<IShape>(z + 2, 0u), z + 1, 2.0);
}
// CHECK: 6.5
// CHECK-NEXT: 3.5
// CHECK-NEXT: 1004.5
// CHECK-NEXT: 1010.5
// CHECK-NEXT: 2005.5
