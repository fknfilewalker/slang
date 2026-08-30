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

/// Return the type for a value whose concrete type is one of the members of `typeSet`: the
/// member itself for a singleton set, otherwise an `UntaggedUnionType` over the set. Every site
/// that materializes such a type must use this function so a singleton has one canonical
/// spelling; otherwise `UntaggedUnionType(TypeSet{T})` and `T` coexist, passes legalize toward
/// different spellings, and the type-flow fixpoint never converges.
IRType* getUntaggedUnionTypeForSet(IRBuilder* builder, IRInst* typeSet);

// Upcast the value in 'arg' to match the destInfo type. This method inserts
// any necessary reinterprets or tag translation instructions.
//
IRInst* upcastSet(IRBuilder* builder, IRInst* arg, IRType* destInfo);

} // namespace Slang
