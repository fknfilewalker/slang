// Backward-differentiate a dynamic dispatch whose callee dispatches dynamically in turn:
// `run` dispatches on `IOuter`, and `OuterA.outer` dispatches again on its `IInner` parameter.
// The inner dispatch makes `OuterA.outer`'s intermediate context type witness-dependent;
// type-flow specialization must give that context one canonical spelling on both sides of the
// apply_bwd/propagate protocol (regression from the auto-diff refactor #9808).

//TEST:INTERPRET(filecheck=CHECK):

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
    float outer(IInner i, float x);
}

public struct OuterA : IOuter
{
    public float k;

    [Differentiable]
    [NoDiffThis]
    public float outer(IInner i, float x) { return i.inner(x) * k; }
}

[Differentiable]
float run(IOuter o, IInner i, float x)
{
    return o.outer(i, x);
}

void main()
{
    IInner i = InnerA(0.5f);
    IOuter o = OuterA(3.0f);

    // run(o, i, x) == 3 * (0.5 * x * x), so the value at x = 2 is 6 and d/dx there is 6 as well.
    var p = diffPair(2.0f);
    bwd_diff(run)(o, i, p, 1.0f);

    //CHECK: value=6.0
    printf("value=%.1f\n", run(o, i, 2.0f));

    //CHECK-NEXT: derivative=6.0
    printf("derivative=%.1f\n", p.d);
}
