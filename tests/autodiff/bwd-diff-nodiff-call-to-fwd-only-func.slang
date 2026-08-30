//TEST:COMPARE_COMPUTE(filecheck-buffer=CHECK):-output-using-type -cpu

// Test that reverse-mode differentiation of a function containing a `no_diff`
// call to a callee that only provides a forward derivative (a
// `[ForwardDerivative(...)]` attribute with no backward counterpart) compiles
// and treats the call as non-differentiable, instead of crashing in the unzip
// step (which can only split calls whose callee is backward-differentiable).
// Forward-mode differentiation of the same function must still use the
// user-provided forward derivative.

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

[ForwardDerivative(fwd_f)]
float f(no_diff float a, float x)
{
    return a * x * x;
}

[Differentiable]
DifferentialPair<float> fwd_f(float a, DifferentialPair<float> x)
{
    return DifferentialPair<float>(a * x.p * x.p, 2.0f * a * x.p * x.d);
}

[Differentiable]
float g(float x)
{
    // `f` is not backward-differentiable, so under `__bwd_diff` this call must
    // contribute only its primal value (the derivative loss is acknowledged by
    // `no_diff`). Under `__fwd_diff` the call is differentiated via `fwd_f`.
    return x * no_diff f(3.0f, x);
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    // Reverse mode: g(x) = x * detach(3x^2), so dg/dx = f(3, 2) = 12.
    var b = diffPair(2.0f, 0.0f);
    __bwd_diff(g)(b, 1.0f);

    // Forward mode: d/dx (x * 3x^2) = 3x^2 + x * 6x = 12 + 24 = 36,
    // with the second term coming from the user-provided `fwd_f`.
    let r = __fwd_diff(g)(diffPair(2.0f, 1.0f));

    outputBuffer[0] = b.d;  // CHECK: 12.0
    outputBuffer[1] = r.p;  // CHECK: 24.0
    outputBuffer[2] = r.d;  // CHECK: 36.0
}
