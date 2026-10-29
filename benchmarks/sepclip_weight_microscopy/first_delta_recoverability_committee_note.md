# First saved delta: what exact arithmetic can recover

Pure analytical committee note; no checkpoint tensors, models, tests or jobs were executed or submitted. Frozen loader/tests/audits remain unchanged. The actual source records D=FP32(B-A), where A is the old parameter and B the first post-normalization parameter, but independently records only the final full model. Current replay computes R=FP32(A+D). The missing independently recorded first model remains a real provenance limitation.

## Exact recovery is conditional, not universally impossible

Assume finite binary32 operands, correctly rounded round-to-nearest/ties-to-even subtraction/addition, gradual underflow, no operand flush and no overflow. Sterbenz's lemma gives exact B-A when A and B have the same sign and their nonzero magnitudes lie within a factor of two: |A|/2 <= |B| <= 2|A|. Because B is already representable, adding this exact D back to A has exact representable result B. Thus R=B. For ordinary normal nonzero weights and modest same-sign updates, exact recovery can hold even though the intermediate was not saved. These conditions are sufficient, not necessary: other differences can also be exact.

The original B is absent, so testing this condition on A and reconstructed R alone does NOT prove it held for A and B. Concrete boundary counterexample under ordinary IEEE arithmetic: A=1, B=the binary32 value immediately below .5, namely .5-2^-25. B-A is exactly halfway between -.5 and the next more-negative binary32 number. Ties-to-even records D=-.5; replay yields R=.5. A and R satisfy the inclusive factor-two condition, while the actual B lies just outside it and is not recovered. Therefore a reconstructed candidate's apparent Sterbenz compatibility cannot be promoted into a full bit certificate.

## A subset can be uniquely certified from observable values under stated arithmetic

Let f_A(b)=RN32(b-A) over finite binary32 candidates. This function is monotone. First require f_A(R)=D. If the nearest finite binary32 candidate below R and the nearest candidate above R both map to values different from D, monotonicity proves R is the UNIQUE finite candidate that could have generated D. One can equivalently count binary32 candidates in the shifted rounding interval A+I(D), with midpoint tie inclusion determined by ties-to-even. This is an observable mathematical uniqueness test, not an independently saved record.

A simple sufficient interior regime follows: R and both adjacent finite neighbors have the same sign as A and all satisfy the Sterbenz factor-two bounds; their subtraction results are exact and distinct, and the backward consistency test gives the unique candidate. Require ordinary IEEE arithmetic and handle signed zero separately. An operation/compiler contract is needed before using this argument as a GPU bit-level assertion; the current source alone does not record the relevant denormal/rounding behavior. No subset count is claimed here because actual artifact scalars were not evaluated.

For contrast, A=2^25 and B=1 produce D=-2^25 and R=0 under ties-to-even. B=0 produces the same saved D. The observed old value and delta cannot distinguish those originals. Cancellation during replay can destroy a small endpoint even when every recorded value is finite. This large illustrative update is a counterexample, not a claimed training event.

## General component error identity and bounds

Define the exact-real subtraction error e_sub=D-(B-A), and observable exact-real replay addition error e_add=R-(A+D). Then:

    R-B = e_sub + e_add

Under the stated IEEE assumptions, let u=2^-24 and eta=2^-149. A conservative all-finite rounding bound is:

    abs(e_sub) <= u*abs(B-A) + eta/2
    abs(e_add) <= u*abs(A+D) + eta/2

Eliminating unknown B from the subtraction bound gives an observable bound:

    abs(R-B) <= abs(e_add) + (u*abs(D)+eta/2)/(1-u)

Here e_add can be computed from the recorded binary32 A,D,R using exact binary rational arithmetic. Ordinary host FP64 sums can lose a tiny addend across a sufficiently large exponent gap, so they are not a universal exact-bit reference. A tighter subtraction bound uses the half-width of D's rounding interval, including asymmetric spacing at binade boundaries and the correct tie endpoints; this also remains observable. If subtraction exactness is established, e_sub is zero and the representable endpoint is recovered. If unique recoverability is established instead, the total reconstruction error is zero; this need not mean either rounding term individually vanished, because the two terms can cancel.

These are absolute component bounds. Relative endpoint error can be arbitrarily large when cancellation makes B small, and relative chord error is ill-scaled for tiny/zero D. For vector parameters, if E_i bounds each component, the aggregate L2 error is at most sqrt(sum E_i^2), with no independence assumption. Summing epoch deltas introduces a new addition rounding term at every step; a small final cumulative reconstruction error does not retrospectively certify the first intermediate because intermediate errors can cancel.

FTZ/DAZ change the premises. If only an operation's subnormal output is flushed, add an absolute error smaller than the smallest normal binary32 magnitude2^-126 for that flush event; the usual eta/2 underflow allowance no longer applies. Flushed subnormal inputs add their own operand errors as well. A complete bound must account for the actual kernel's input/output flush events, rounding mode and overflow behavior, rather than silently assert the IEEE bound. Sterbenz cancellation into subnormals needs gradual underflow or a separately validated exact operation. If relevant operands/results are normal, underflow flush concerns can be excluded for those operations; this must be demonstrated, not assumed for the full bank.

Signed zero has two encodings with the same real value. Even exact real subtraction/addition need not retain the original endpoint zero sign, so real-value uniqueness does not establish every bit unless signed-zero behavior is explicitly checked. NaN/Infinity and overflow are outside these finite bounds and require withholding, as the existing replay guards do.

## Consequence for actual first-effect reporting

The existing report label is correct: first bank is old plus saved FP32 first delta; the original first model is not independently recorded. Under verified arithmetic, observable unique-recovery tests could establish exact nonzero components and absolute bounds could quantify the remainder, without inventing missing logits or a full-bank certificate. This would strengthen reconstruction evidence, not make the record independent. No frozen code change or extra job is proposed here.

A small parameter reconstruction bound alone supplies no numerical physical-return bound: policy sensitivity and closed-loop dynamics would need separate control. Actual first-effect probes should retain the recording label, expose available reconstruction precision, and use their executed architecture-specific policy/physics contracts. The exact final model remains a separate recorded endpoint reference.
