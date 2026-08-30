// Backward-differentiate a dynamically dispatched method that takes `inout` parameters -- a
// non-differentiable one (an RNG-style state) and a differentiable one -- and dispatches again
// internally, so its intermediate context type is witness-dependent.

//TEST:INTERPRET(filecheck=CHECK):

public struct Rng
{
    public uint state;
    [mutating] public float next() { state = state * 3u + 1u; return float(state % 4u); }
}

public dyn interface IInner
{
    [Differentiable]
    float inner(float x);
}

public struct InnerA : IInner
{
    public float a;
    [Differentiable]
    [NoDiffThis]
    public float inner(float x) { return a * x * x; }
}

public dyn interface IOuter
{
    [Differentiable]
    float outer(IInner i, inout Rng rng, inout float acc, float x);
}

public struct OuterA : IOuter
{
    public float k;
    [Differentiable]
    [NoDiffThis]
    public float outer(IInner i, inout Rng rng, inout float acc, float x)
    {
        // rng.next() is non-differentiable and consumes state; it must be replayed from the
        // same pre-call state on the backward pass to reproduce the same scale.
        float scale = no_diff rng.next();
        acc = acc * k + x;
        return i.inner(x) * scale + acc;
    }
}

[Differentiable]
float run(IOuter o, IInner i, inout Rng rng, inout float acc, float x)
{
    return o.outer(i, rng, acc, x);
}

void main()
{
    IInner i = InnerA(0.5f);
    IOuter o = OuterA(3.0f);

    // Rng state 5: next() -> state 16, returns 0. Use state 6: -> 19, returns 3.
    Rng rng; rng.state = 6u;
    float acc0 = 1.0f;

    // acc' = 3 * 1 + x = 5 (at x = 2); result = 0.5 * x * x * 3 + acc' = 6 + 5 = 11.
    // d/dx = 0.5 * 2 * x * 3 + 1 = 7 at x = 2; d/dacc0 = 3.
    var px = diffPair(2.0f);
    var pacc = diffPair(acc0);
    Rng rngPrimal = rng;
    float accPrimal = acc0;
    float value = run(o, i, rngPrimal, accPrimal, 2.0f);
    bwd_diff(run)(o, i, rng, pacc, px, 1.0f);

    //CHECK: value=11.0
    printf("value=%.1f\n", value);
    //CHECK-NEXT: dx=7.0 dacc=3.0
    printf("dx=%.1f dacc=%.1f\n", px.d, pacc.d);
}
