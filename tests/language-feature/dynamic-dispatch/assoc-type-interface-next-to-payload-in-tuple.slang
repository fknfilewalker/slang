// A dispatched requirement takes `Tuple<Ctx, IA>`. The `Ctx` element crosses the dispatch as a
// payload, while the `IA` element is an ordinary existential that each call site specializes on.
// Two call sites dispatch over overlapping sets of implementations, {S0, S1} with an `A0` and
// {S2, S0} with an `A1`, so `S0.use` is reached from both. Each implementation in a dispatched set
// must still see the same `IA` binding at a call site, or the dispatcher's parameter and the
// implementation's parameter disagree on the `IA` element.

//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-cpu -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

interface IA { float f(); }
struct A0 : IA { float v; float f() { return v; } }
struct A1 : IA { float v; float f() { return 10 * v; } }

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

IShape pickA(uint i)
{
    IShape s;
    if (i == 0)
        s = S0();
    else
        s = S1();
    return s;
}

IShape pickB(uint i)
{
    IShape s;
    if (i == 0)
        s = S2();
    else
        s = S0();
    return s;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    IA a0 = A0(1);
    IA a1 = A1(1);
    outputBuffer[0] = run(pickA(tid.x + 0), 2.0, a0);
    outputBuffer[1] = run(pickA(tid.x + 1), 2.0, a0);
    outputBuffer[2] = run(pickB(tid.x + 0), 2.0, a1);
    outputBuffer[3] = run(pickB(tid.x + 1), 2.0, a1);
}

// S0 + A0: 2 + 1 = 3
// CHECK: 3
// S1 + A0: 2 + 1 + 1 = 4
// CHECK-NEXT: 4
// S2 + A1: 200 + 10 = 210
// CHECK-NEXT: 210
// S0 + A1: 2 + 10 = 12
// CHECK-NEXT: 12
