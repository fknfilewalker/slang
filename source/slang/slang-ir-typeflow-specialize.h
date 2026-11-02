// slang-ir-typeflow-specialize.h
#pragma once
#include "slang-ir.h"
#include "slang-target-program.h"

namespace Slang
{
struct SpecializationContext;

// Convert dynamic insts such as `LookupWitnessMethod`, `ExtractExistentialValue`,
// `ExtractExistentialType`, `ExtractExistentialWitnessTable` and more into specialized versions
// based on the possible values at at the use sites, based on a data-flow-style interprocedural
// analysis.
//
// This pass is intended to be run after all specialization insts with concrete arguments have
// already been processed.
//
// This pass may generate more `Specialize` insts, so it should be run in a loop with
// the standard specialization pass until a no more changes can be made.
//
bool specializeDynamicInsts(
    IRModule* module,
    TargetProgram* target,
    DiagnosticSink* sink,
    SpecializationContext* context,
    bool shouldReportDynamicDispatchSites);

bool isSetSpecializedGeneric(IRInst* callee);

IROp getSetOpFromType(IRType* type);

// Return the type of the payload that carries a value of the declared type `type` through a type
// variable, such as an associated type `T.Ctx` whose witness table is chosen at run time. If `type`
// is concrete, that is `type` itself. Otherwise each non-COM interface in `type` is replaced by the
// tagged union over all of its witness tables; an `Optional` of an interface also includes the
// `none` witness.
//
// Consider `Sphere.Ctx = Tuple<IBsdf, float>`, where `Diffuse` and `Plastic` conform to `IBsdf`.
// A caller that sees only `T.Ctx` passes the value as an AnyValue, which `Sphere.make` packs and
// `Sphere.use` unpacks. The payload type is `Tuple(TaggedUnion({Diffuse, Plastic}), float)`. It
// depends only on the declared type, so the producer, the consumer, the caller and a generic
// specialized with a set of witness tables all agree on it. Type-flow uses it as the type an
// associated-type lookup selects, and the generic specializer uses it when it folds a lookup over a
// witness-table set.
//
// Returns nullptr if `type` holds an interface, outside an `Optional`, that has no witness table,
// because no value of `type` exists. That interface is written to `outInterfaceWithoutTables`
// unless it is null. Also returns nullptr if `type` holds a non-concrete type that type-flow does
// not refine, such as `Conditional<IFoo, true>`; that type is written to `outUnsupportedType`
// unless it is null.
//
IRType* getTypeVariablePayloadType(
    IRModule* module,
    IRType* type,
    IRInterfaceType** outInterfaceWithoutTables = nullptr,
    IRType** outUnsupportedType = nullptr);
} // namespace Slang
