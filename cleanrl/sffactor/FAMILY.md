# Successor-Feature Factorized Critic

`V(s) = psi(s) . w` factorization of the IterThink v24 Beta critic head:
vector TD(lambda) successor features + learned linear reward weights, plus
the scalar-MSE and symlog-MSE falsifier controls the SF line was tested
against. (`sffactor_rewardanchor` RNG-paired variants included.)
