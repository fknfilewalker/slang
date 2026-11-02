// A dynamically dispatched requirement returns or takes `Optional<Ctx>`, and one implementation
// binds the associated type `Ctx` to an interface. Its `Optional<IA>` is an existential rather than
// an optional of a payload, so the value cannot cross the dispatch, and the program is rejected
// with error 52011 instead of an internal error.

//DIAGNOSTIC_TEST:SIMPLE(diag=CHECK): -target cpp -stage compute -entry computeMain -conformance "A0:IA=0" -conformance "B0:IB=0" -conformance "S0:IShape=0" -conformance "S1:IShape=1" -conformance "U0:IUser=0" -conformance "U1:IUser=1"

interface IA { float fa(float x); }
struct A0 : IA { float k; float fa(float x) { return k + x; } }

interface IB { float fb(float x); }
struct B0 : IB { float k; float fb(float x) { return k * x; } }

// The requirement returns `Optional<Ctx>`.
interface IShape
{
    associatedtype Ctx;
    Optional<Ctx> makeOpt(float x);
    float use(Ctx c);
}

struct S0 : IShape
{
    typealias Ctx = IA;
    Optional<Ctx> makeOpt(float x) { return createDynamicObject<IA>(0, x); }
    float use(Ctx c) { return c.fa(1); }
}

struct S1 : IShape
{
    typealias Ctx = float;
    Optional<Ctx> makeOpt(float x) { return x; }
    float use(Ctx c) { return c; }
}

float runShape<S : IShape>(S s, float x)
{
    let o = s.makeOpt(x);
//CHECK:             ^ optional of an interface-bound associated type incompatible with dynamic dispatch
//CHECK:             ^ an 'Optional' of an associated type bound to interface 'IA' cannot cross a dynamic dispatch
    if (o.hasValue)
        return s.use(o.value);
    return -1;
}

// The requirement takes `Optional<Ctx>`.
interface IUser
{
    associatedtype Ctx;
    Ctx make(float x);
    float useOpt(Optional<Ctx> c);
}

struct U0 : IUser
{
    typealias Ctx = IB;
    Ctx make(float x) { return createDynamicObject<IB>(0, x); }
    float useOpt(Optional<Ctx> c) { return c.hasValue ? c.value.fb(2) : 0; }
//CHECK:                       ^ optional of an interface-bound associated type incompatible with dynamic dispatch
//CHECK:                       ^ an 'Optional' of an associated type bound to interface 'IB' cannot cross a dynamic dispatch
}

struct U1 : IUser
{
    typealias Ctx = float;
    Ctx make(float x) { return x; }
    float useOpt(Optional<Ctx> c) { return c.hasValue ? c.value : 0; }
}

float runUser<U : IUser>(U u, float x)
{
    Optional<U.Ctx> o = u.make(x);
    return u.useOpt(o);
}

uniform uint dynId;
RWStructuredBuffer<float> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain()
{
    IShape s = createDynamicObject<IShape>(dynId, 0u);
    outputBuffer[0] = runShape(s, 1.0);
    IUser u = createDynamicObject<IUser>(dynId, 0u);
    outputBuffer[1] = runUser(u, 1.0);
}
