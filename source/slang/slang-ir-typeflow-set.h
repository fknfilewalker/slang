// slang-ir-typeflow-set.h
#pragma once
#include "slang-ir-insts.h"
#include "slang-ir.h"

namespace Slang
{

//
// Helpers to iterate over elements of a collection.
//

template<typename F>
void forEachInSet(IRModule* module, IRSetBase* info, F func)
{
    List<IRInst*>& elements = *module->getContainerPool().getList<IRInst>();

    for (UInt i = 0; i < info->getOperandCount(); ++i)
        elements.add(info->getElement(i));

    for (auto element : elements)
        func(element);

    module->getContainerPool().free(&elements);
}

// Upcast the value in 'arg' to match the destInfo type. This method inserts
// any necessary reinterprets or tag translation instructions.
//
IRInst* upcastSet(IRBuilder* builder, IRInst* arg, IRType* destInfo);

// Return `arg` converted to `destType` by unpacking each untagged-union value inside it into the
// type at the same position of `destType`. This is the inverse of the packing that `upcastSet`
// does. It looks through tuples, arrays and optionals, which is where a value of a type variable
// can sit inside a parameter.
//
// Consider a requirement `float use(Tuple<Ctx, float> c)` called through a dynamically dispatched
// `IShape`. The dispatcher receives `Tuple(UntaggedUnion({...}), float)`, and `Sphere.use` with
// `Ctx = Tuple<IBsdf, float>` takes `Tuple(Tuple(TaggedUnion(...), float), float)`, so the first
// element is unpacked and the second one is passed on.
//
IRInst* unpackPayloads(IRBuilder* builder, IRInst* arg, IRType* destType);

} // namespace Slang
