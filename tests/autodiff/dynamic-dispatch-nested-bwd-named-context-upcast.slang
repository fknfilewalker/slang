// Differentiate backward through a dispatched method whose minimal context holds an existential.
// `Plane.shadeAll` calls the differentiable generic `helper`, which passes the no_diff
// `Tuple<Plane.Ctx, IBsdf>` to `Plane.shade`, and `Plane.Ctx` is `Tuple<int, IBsdf>`. With debug
// info the specialized `Tuple<int, Diffuse>` stays live in the minimal context of
// `Plane.shadeAll`, so type-flow upcasts that named context tuple to the tagged-union form of its
// `IShape` dispatch signature element by element. The trailing name sentinel of the named tuple
// is not an element and must not be extracted.

//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-cpu -compute -shaderobj -output-using-type -Xslang -g

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

[anyValueSize(16)]
interface IBsdf
{
    [Differentiable] float eval(float x);
}

struct Diffuse : IBsdf
{
    float k;
    [Differentiable, NoDiffThis] float eval(float x) { return k * x * x; }
}

struct Plastic : IBsdf
{
    float k;
    [Differentiable, NoDiffThis] float eval(float x) { return k * x * x * x; }
}

[anyValueSize(64)]
interface IShape
{
    associatedtype Ctx;
    Ctx make(float a);
    [Differentiable] float shade(no_diff Tuple<Ctx, IBsdf> c, float x);
    [Differentiable] float shadeAll(no_diff float a, IBsdf b, float x);
}

[Differentiable]
float helper<S : IShape>(S s, no_diff float a, IBsdf b, float x)
{
    let c = no_diff s.make(a);
    return s.shade(makeTuple<S.Ctx, IBsdf>(c, b), x);
}

struct Sphere : IShape
{
    typealias Ctx = float;
    Ctx make(float a) { return a; }

    [Differentiable, NoDiffThis]
    float shade(no_diff Tuple<Ctx, IBsdf> c, float x)
    {
        IBsdf b = c._1;
        return c._0 * b.eval(x);
    }

    [Differentiable, NoDiffThis]
    float shadeAll(no_diff float a, IBsdf b, float x) { return helper(this, a, b, x); }
}

struct Plane : IShape
{
    typealias Ctx = Tuple<int, IBsdf>;

    Ctx make(float a)
    {
        IBsdf d = Diffuse(1);
        return makeTuple<int, IBsdf>(int(a) * 2, d);
    }

    [Differentiable, NoDiffThis]
    float shade(no_diff Tuple<Ctx, IBsdf> c, float x)
    {
        var i = c._0;
        IBsdf b0 = i._1;
        IBsdf b1 = c._1;
        return float(i._0) * b0.eval(x) + b1.eval(x) + x;
    }

    [Differentiable, NoDiffThis]
    float shadeAll(no_diff float a, IBsdf b, float x) { return helper(this, a, b, x); }
}

[Differentiable]
float outerDyn(IShape s, no_diff float a, IBsdf b, float x) { return s.shadeAll(a, b, x); }

float bwdDyn(IShape s, float a, IBsdf b, float x)
{
    var dp = diffPair(x, 0.0f);
    bwd_diff(outerDyn)(s, a, b, dp, 1.0f);
    return dp.d;
}

IShape pickA(uint i)
{
    IShape s;
    if (i == 0)
        s = Sphere();
    else
        s = Plane();
    return s;
}

IShape pickB(uint i)
{
    IShape s;
    if (i == 0)
        s = Plane();
    else
        s = Sphere();
    return s;
}

IBsdf pickBsdf(uint i, float k)
{
    IBsdf b;
    if (i == 0)
        b = Diffuse(k);
    else
        b = Plastic(k);
    return b;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 tid: SV_DispatchThreadID)
{
    uint z = tid.x;
    IBsdf d2 = pickBsdf(z, 2);
    IBsdf p3 = pickBsdf(z + 1, 3);

    // Sphere, a = 2, Diffuse(2): 2 * 2x^2, derivative 8x at x = 1.
    // CHECK: 8
    outputBuffer[0] = bwdDyn(pickA(z + 0), 2.0, d2, 1.0);

    // Plane, a = 2, Plastic(3): 4 * x^2 + 3x^3 + x, derivative 8x + 9x^2 + 1 at x = 1.
    // CHECK-NEXT: 18
    outputBuffer[1] = bwdDyn(pickA(z + 1), 2.0, p3, 1.0);

    // CHECK-NEXT: 18
    outputBuffer[2] = bwdDyn(pickB(z + 0), 2.0, p3, 1.0);

    // Sphere, a = 3, Diffuse(2): 3 * 2x^2, derivative 12x at x = 2.
    // CHECK-NEXT: 24
    outputBuffer[3] = bwdDyn(pickB(z + 1), 3.0, d2, 2.0);
}
