// A generic type specialized with an interface type cannot be used as another interface it
// conforms to, because its witness table cannot record which concrete type the argument holds.

//DIAGNOSTIC_TEST:SIMPLE(diag=CHECK):-target spirv -stage compute -entry computeMain
//DIAGNOSTIC_TEST:SIMPLE(diag=CHECK):-target hlsl -stage compute -entry computeMain
//DIAGNOSTIC_TEST:SIMPLE(diag=CHECK):-target glsl -stage compute -entry computeMain
//DIAGNOSTIC_TEST:SIMPLE(diag=CHECK):-target metal -stage compute -entry computeMain

interface IFoo
{
    float eval();
}

struct A : IFoo
{
    float x;
    float eval() { return x; }
}

struct C : IFoo
{
    float4 x;
    float eval() { return x.w; }
}

interface IBar
{
    float run();
}

struct Wrap<T : IFoo> : IBar
{
    T inner;
    float run() { return inner.eval(); }
}

struct Other : IBar
{
    float a;
    float run() { return a; }
}

IFoo makeFoo(int i)
{
    if (i == 0)
        return A(1.0);
    return C(float4(1, 2, 3, 4));
}

float callThroughInterface<T : IFoo>(T t)
{
    Wrap<T> w = { t };
    IBar b = w;
//CHECK:     ^ E33180
//CHECK:     ^ specializing 'Wrap' with an existential type
    return b.run();
}

IBar makeBar<T : IFoo>(T t, int j)
{
    if (j == 0)
    {
        Wrap<T> w = { t };
        return w;
//CHECK:       ^ E33180
//CHECK:       ^ specializing 'Wrap' with an existential type
    }
    return Other(2.0);
}

float runBar<B : IBar>(B b)
{
    return b.run();
}

float callThroughGeneric<T : IFoo>(T t)
{
    Wrap<T> w = { t };
    return runBar(w);
//CHECK:         ^ E33180
//CHECK:         ^ specializing 'Wrap' with an existential type
}

RWStructuredBuffer<float> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    IFoo foo = makeFoo(tid.x);
    outputBuffer[0] = callThroughInterface(foo);
    outputBuffer[1] = makeBar(foo, tid.y).run();
    outputBuffer[2] = callThroughGeneric(foo);
}
