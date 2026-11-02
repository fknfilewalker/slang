// A dispatched requirement takes `Tuple<Ctx, IA>`, where every implementation binds the associated
// type `Ctx` to a concrete type. The tuple is not concrete because of `IA`, so the `Ctx` element
// crosses the dispatch as a payload, and each implementation must see its own concrete element
// type there rather than the caller's packed `AnyValue`.

//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-cpu -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

interface IA { float f(); }
struct A0 : IA { float v; float f() { return v; } }

interface IShape
{
    associatedtype Ctx;
    Ctx make(float x);
    float use(Tuple<Ctx, IA> c);
}

struct S0 : IShape
{
    typealias Ctx = float;
    Ctx make(float x) { return x; }
    float use(Tuple<Ctx, IA> c) { return c._0 + c._1.f(); }
}
struct S1 : IShape
{
    typealias Ctx = float2;
    Ctx make(float x) { return float2(x, 1); }
    float use(Tuple<Ctx, IA> c) { return c._0.x + c._0.y + c._1.f(); }
}
struct S2 : IShape
{
    typealias Ctx = int;
    Ctx make(float x) { return int(x) * 100; }
    float use(Tuple<Ctx, IA> c) { return float(c._0) + c._1.f(); }
}

float run<T : IShape>(T s, float x, IA a)
{
    let c = s.make(x);
    return s.use(makeTuple<T.Ctx, IA>(c, a));
}

[numthreads(1, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    IA a0 = A0(1);
    outputBuffer[0] = run(createDynamicObject<IShape>(tid.x + 0, 0), 2.0, a0);
    outputBuffer[1] = run(createDynamicObject<IShape>(tid.x + 1, 0), 2.0, a0);
    outputBuffer[2] = run(createDynamicObject<IShape>(tid.x + 2, 0), 2.0, a0);
}
// S0 + A0: 2 + 1 = 3
// CHECK: 3
// S1 + A0: 2 + 1 + 1 = 4
// CHECK-NEXT: 4
// S2 + A0: 200 + 1 = 201
// CHECK-NEXT: 201

//TEST_INPUT: type_conformance S0:IShape = 0
//TEST_INPUT: type_conformance S1:IShape = 1
//TEST_INPUT: type_conformance S2:IShape = 2
