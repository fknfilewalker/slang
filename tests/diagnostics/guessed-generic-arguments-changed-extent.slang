//DIAGNOSTIC_TEST:SIMPLE(diag=CHECK):

// Outside function bodies the parser guesses whether `<` starts generic arguments, and the guess
// also decides where an initializer ends: `a < b, y = c > (d)` was read as one generic call that
// takes `y` into `x`. Checking parses `x`'s initializer again and finds that the comparison ends
// at the comma, so this is reported instead of silently dropping `y`.

static const int a = 1, b = 2, c = 3, d = 4;
static const bool x = a < b, y = c > (d);
//CHECK:                   ^ ambiguous '<' outside a function body
//CHECK:                   ^ this expression was read as generic arguments when the declaration was parsed, which made it extend too far; put parentheses around the comparison
