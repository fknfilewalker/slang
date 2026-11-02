// The backward derivative of `test` saves an existential (`saved`) in its minimal context, and
// `test` is differentiated at two static call sites with different concrete types. Type-flow
// specializes the context per call site, so each call site keeps its own precise context type.
// The custom derivative of the `[ForceInline]` extension method `load` runs on the saved value.

//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-cpu -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

public interface IFoo
{
    float read(uint id);
    void write(uint id, float value);
}

public extension<Foo : IFoo> Foo
{
    [ForceInline]
    [Differentiable]
    [BackwardDerivative(_load_bwd_indices)]
    public float load(int i0)
    {
        return this.read(i0);
    }

    public void _load_bwd_indices(int i0, float dOut)
    {
        write(i0, dOut);
    }
}

struct FooImpl : IFoo
{
    float read(uint id) { return outputBuffer[0]; }
    void write(uint id, float value) { outputBuffer[0] = value; }
}
struct BarImpl : IFoo
{
    float read(uint id) { return outputBuffer[1]; }
    void write(uint id, float value) { outputBuffer[1] = value; }
}

// Side effect, so the result is checkpointed instead of recomputed.
IFoo pick(IFoo f) { outputBuffer[4] += 1; return f; }

[Differentiable]
float test(IFoo foo)
{
    IFoo saved = no_diff pick(foo);
    return (saved.load(0) + 2) * (saved.load(0) + 2);
}

[numthreads(1,1,1)]
void computeMain(uint tid : SV_DispatchThreadID)
{
    FooImpl foo;
    bwd_diff(test)(foo, 1.0);
    BarImpl bar;
    bwd_diff(test)(bar, 1.0);
}

// d/dx (x + 2)^2 at x = 0 reaches each `load` as 2, and `write` stores it; `pick` runs once per
// derivative call because its result is checkpointed rather than recomputed.
// CHECK: 2.0
// CHECK-NEXT: 2.0
// CHECK-NEXT: 0.0
// CHECK-NEXT: 0.0
// CHECK-NEXT: 2.0
