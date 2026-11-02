// Dispatched requirements take an associated type `Ctx` by `__ref` and `__constref`, at the top
// level and nested in `Tuple<Ctx, float>`. Each implementation binds `Ctx` to a concrete type, so
// the interface holds the value packed while the implementation holds it unpacked. The dispatch
// wrapper must initialize the implementation's copy from the argument for both directions, and
// write it back after the call for `__ref` only.

//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-cpu -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

interface IShape
{
    associatedtype Ctx;
    Ctx make(float x);
    float readRef(__ref Tuple<Ctx, float> c);
    float readConstRef(__constref Tuple<Ctx, float> c);
    float readRefTop(__ref Ctx c);
}

struct S0 : IShape
{
    typealias Ctx = float;
    Ctx make(float x) { return x; }
    float readRef(__ref Tuple<Ctx, float> c) { c._1 = c._1 + 1; return c._0 + c._1; }
    float readConstRef(__constref Tuple<Ctx, float> c) { return c._0 * c._1; }
    float readRefTop(__ref Ctx c) { c = c + 1; return c; }
}

struct S1 : IShape
{
    typealias Ctx = float2;
    Ctx make(float x) { return float2(x, 10); }
    float readRef(__ref Tuple<Ctx, float> c) { c._1 = c._1 + 1; return c._0.x + c._0.y + c._1; }
    float readConstRef(__constref Tuple<Ctx, float> c) { return (c._0.x + c._0.y) * c._1; }
    float readRefTop(__ref Ctx c) { c.x = c.x + 1; return c.x + c.y; }
}

float runRef<T : IShape>(T s, float x)
{
    var t = makeTuple<T.Ctx, float>(s.make(x), 3.0);
    let r = s.readRef(t);
    return r * 1000 + t._1;
}

float runConstRef<T : IShape>(T s, float x)
{
    let t = makeTuple<T.Ctx, float>(s.make(x), 3.0);
    return s.readConstRef(t);
}

float runRefTop<T : IShape>(T s, float x)
{
    var c = s.make(x);
    let first = s.readRefTop(c);
    return first * 100 + s.readRefTop(c);
}

//TEST_INPUT: type_conformance S0:IShape = 0
//TEST_INPUT: type_conformance S1:IShape = 1

[numthreads(1, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    outputBuffer[0] = runRef(createDynamicObject<IShape>(tid.x + 0, 0), 2.0);
    outputBuffer[1] = runRef(createDynamicObject<IShape>(tid.x + 1, 0), 2.0);
    outputBuffer[2] = runConstRef(createDynamicObject<IShape>(tid.x + 0, 0), 2.0);
    outputBuffer[3] = runConstRef(createDynamicObject<IShape>(tid.x + 1, 0), 2.0);
    outputBuffer[4] = runRefTop(createDynamicObject<IShape>(tid.x + 0, 0), 2.0);
    outputBuffer[5] = runRefTop(createDynamicObject<IShape>(tid.x + 1, 0), 2.0);
}
// S0 readRef: c = (2, 3) -> c._1 = 4, r = 6, t._1 = 4: 6004
// CHECK: 6004
// S1 readRef: c = ((2, 10), 3) -> c._1 = 4, r = 16, t._1 = 4: 16004
// CHECK-NEXT: 16004
// S0 readConstRef: 2 * 3 = 6
// CHECK-NEXT: 6
// S1 readConstRef: 12 * 3 = 36
// CHECK-NEXT: 36
// S0 readRefTop: c = 2 -> 3, then 4: 304
// CHECK-NEXT: 304
// S1 readRefTop: c = (2, 10) -> 3 + 10 = 13, then 14: 1314
// CHECK-NEXT: 1314
