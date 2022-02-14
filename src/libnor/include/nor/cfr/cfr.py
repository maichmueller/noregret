def cfr(
        h=[],    # history
        π_i={},  # a mapping from player_id to π_i(h)
        i=0,     # player_id
        t=0      # timestep
):
    # here I skip the chance node case for simplicity
    if h is terminal:
        z = h
        return u(i, z)  # utility u(z) for player i

    # information set of the history, h ∈ I
    I = h.I

    # R[t] is counterfactual regret (without maximum), R^t
    # σ[t] is the strategy from regret matching at time t, σ^t
    σ[t][I] = regret_matching(R[t][I])

    # counterfactual value divided by π_{-i}(h), v(σ|I -> a) / π_{-i}(h)
    # I use `**` dict unpack syntax like here https://www.codespeedy.com/merge-two-dictionaries-in-python-update-double-star/
    v[I] = {a: cfr(h + [a], {**π_i, P(h): π_i[P(h)] * σ[t][I][a]}, i, t)
            for a in A[I]}

    # counterfactual value v(σ) divided by π_{-i}(h)
    # expectation of v(σ|I -> a) / π_{-i}(h) over actions
    vI = sum(σ[t][I][a] * via for a,via in v[I].items())

    for a in A[I]:
        # π_{-i}(h), probability of reaching h when only
        # considering other players
        π_neg_i_h = prod(prob for pid, prob in π_i.items() if pid != P(h))

        # counterfactual regret at time t + 1
        R[t+1][I][a] += π_neg_i_h * (v[I][a] - vI)

        # strategy average over time
        σ_avg[I, a] += π_i[P(h)] * σ[t][I][a]
    return vI

# using
for iteration in range(10000):  # 10000 iterations
    for player_id in range(2):    # 2 player
        cfr(i=player_id, t=iteration)