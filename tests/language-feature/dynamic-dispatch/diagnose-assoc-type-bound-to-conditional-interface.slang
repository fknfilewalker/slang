// An associated type bound to `Conditional<IA, true>` crosses a dynamically dispatched call as a
// payload, but type-flow has no info for the values of a `Conditional` that holds an interface,
// so it cannot choose a payload type and rejects the program with error 41014.

//DIAGNOSTIC_TEST:SIMPLE(diag=CHECK): -target cpp -stage compute -entry computeMain -conformance "A0:IA=0" -conformance "A1:IA=1" -conformance "S0:IShape=0" -conformance "S1:IShape=1"

interface IA { float fa(float x); }
struct A0 : IA { float k; float fa(float x) { return k + x; } }
struct A1 : IA { float k; float fa(float x) { return k * x; } }

interface IShape
{
    associatedtype Ctx;
    Ctx make(float x);
    float use(Ctx c);
}

struct S0 : IShape
{
    typealias Ctx = Conditional<IA, true>;
    Ctx make(float x) { return Ctx(createDynamicObject<IA>(0, x)); }
    float use(Ctx c) { return 1; }
}

struct S1 : IShape
{
    typealias Ctx = float;
    Ctx make(float x) { return x; }
    float use(Ctx c) { return c; }
}

float runGeneric<S : IShape>(S s, float x)
{
    let c = s.make(x);
//CHECK:^ type cannot be packed for dynamic dispatch
//CHECK:^ type 'Conditional<IA,true>' contains fields that cannot be packed into ordinary bytes for dynamic dispatch.
    return s.use(c);
}

uniform uint dynId;
RWStructuredBuffer<float> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain()
{
    IShape s = createDynamicObject<IShape>(dynId, 0u);
    outputBuffer[0] = runGeneric(s, 1.0);
}
