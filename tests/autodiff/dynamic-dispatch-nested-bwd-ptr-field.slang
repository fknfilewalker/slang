// Nested dynamic dispatch in reverse mode where the inner existential is an interface-pointer
// field of the outer implementation rather than a parameter: `IShape::store` is dispatched
// dynamically, and `Mesh::store` dispatches again through its `IFilm*` field. The outer callee's
// signature shows no existential at all, so the witness-dependence of its context is visible
// only through the stored field.

//TEST:COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -output-using-type -emit-spirv-directly
//TEST:COMPARE_COMPUTE(filecheck-buffer=CHECK):-cuda -output-using-type

//TEST_INPUT: type_conformance FilmA:IFilm = 1
//TEST_INPUT: type_conformance FilmB:IFilm = 2
//TEST_INPUT: type_conformance Mesh:IShape = 1
//TEST_INPUT: type_conformance Quad:IShape = 2

[anyValueSize(16)]
interface IFilm
{
    [Differentiable]
    float add(float x);
}

struct FilmA : IFilm
{
    float s;
    [Differentiable]
    [NoDiffThis]
    float add(float x) { return s * x * x; }
}

struct FilmB : IFilm
{
    float s;
    [Differentiable]
    [NoDiffThis]
    float add(float x) { return s * x * x * x; }
}

[anyValueSize(16)]
interface IShape
{
    [Differentiable]
    float store(float x);
}

struct Mesh : IShape
{
    IFilm* film;
    [Differentiable]
    [NoDiffThis]
    float store(float x) { return film->add(x) * 2.0f; }
}

struct Quad : IShape
{
    IFilm* film;
    [Differentiable]
    [NoDiffThis]
    float store(float x) { return film->add(x) + 1.0f; }
}

[Differentiable]
float run(IShape s, float x)
{
    return s.store(x);
}

// Serialized existential: FilmA (type id 1) with s = 0.5 as its payload.
//TEST_INPUT: set film = ubuffer(data=[0 0 1 0 0x3f000000 0 0 0], stride=4)
uniform IFilm* film;

//TEST_INPUT: set outBuffer = out ubuffer(data=[0 0 0 0], stride=4)
RWStructuredBuffer<float> outBuffer;

[shader("compute"), numthreads(1, 1, 1)]
void computeMain(uint3 id : SV_DispatchThreadID)
{
    // Build the outer object on the device so its pointer field is valid.
    Mesh m;
    m.film = film;
    IShape s = m;

    // run == 2 * (0.5 * x * x), so at x = 2 the value is 4 and d/dx is 4.
    var p = diffPair(2.0f);
    bwd_diff(run)(s, p, 1.0f);

    // CHECK: 4.0
    outBuffer[0] = run(s, 2.0f);
    // CHECK-NEXT: 4.0
    outBuffer[1] = p.d;
}
