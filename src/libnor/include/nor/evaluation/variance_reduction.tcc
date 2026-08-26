
#ifndef NOR_EVALUATION_VARIANCE_REDUCTION_TCC
   #define NOR_EVALUATION_VARIANCE_REDUCTION_TCC

// NOTE: included at the bottom of variance_reduction.hpp, i.e. OUTSIDE its
// namespaces, hence the explicit wrapping here.
namespace nor {
namespace evaluation {

///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////// game tree ///////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

template < typename Env, typename Policy, typename ActionPolicy >
   requires concepts::fosg< std::remove_cvref_t< Env > > and (concepts::state_policy_no_default< Policy, auto_info_state_type< std::remove_cvref_t< Env > >, auto_action_type< std::remove_cvref_t< Env > >, ActionPolicy > or concepts::state_policy_view< Policy, auto_info_state_type< std::remove_cvref_t< Env > >, auto_action_type< std::remove_cvref_t< Env > > >)
ProfileGameTree< Env, Policy, ActionPolicy >::ProfileGameTree(
   const Env& env,
   const world_state_type& root_state,
   const player_hashmap< Policy >& profile
)
{
   m_root_roster = env.players(root_state);
   for(auto player : m_root_roster | utils::is_actual_player_filter) {
      m_players.push_back(player);
   }
   m_nodes.reserve(1024);

   // the root is always the first node created (index 0)
   _build(
      env,
      utils::static_unique_ptr_downcast< world_state_type >(utils::clone_any_way(root_state)),
      1.,
      std::invoke([&] {
         player_hashmap< std::vector< std::pair< observation_type, observation_type > > > obs_map{};
         for(auto player : m_players) {
            obs_map.try_emplace(player);
         }
         return obs_map;
      }),
      std::invoke([&] {
         player_hashmap< info_state_type > infostates{};
         for(auto player : m_players) {
            infostates.emplace(player, info_state_type{player});
         }
         return infostates;
      }),
      profile
   );

   assert(not m_nodes.empty() and m_nodes.front().reach_probability == 1.);
}

template < typename Env, typename Policy, typename ActionPolicy >
   requires concepts::fosg< std::remove_cvref_t< Env > > and (concepts::state_policy_no_default< Policy, auto_info_state_type< std::remove_cvref_t< Env > >, auto_action_type< std::remove_cvref_t< Env > >, ActionPolicy > or concepts::state_policy_view< Policy, auto_info_state_type< std::remove_cvref_t< Env > >, auto_action_type< std::remove_cvref_t< Env > > >)
size_t ProfileGameTree< Env, Policy, ActionPolicy >::_build(
   const env_type& env,
   uptr< world_state_type > state,
   double reach_probability,
   player_hashmap< std::vector< std::pair< observation_type, observation_type > > >
      observation_buffer,
   player_hashmap< info_state_type > infostate_map,
   const player_hashmap< Policy >& profile
)
{
   // NOTE: no references into m_nodes may be held across the recursive calls
   // below -- they append to m_nodes and thereby invalidate them. All node data
   // is staged in locals and committed to m_nodes[this_idx] at the very end.
   const size_t this_idx = m_nodes.size();
   m_nodes.emplace_back();
   m_nodes.back().reach_probability = reach_probability;

   if(env.is_terminal(*state)) {
      // NOTE: pass the ROOT participant roster explicitly so that players who
      // folded out of poker-like games still contribute their sunk stakes
      auto rewards = rm::collect_rewards(env, *state, m_root_roster);
      m_nodes[this_idx].is_terminal = true;
      m_nodes[this_idx].rewards = rm::PlayerValueTable{rewards};
      m_nodes[this_idx].values = m_nodes[this_idx].rewards;
      return this_idx;
   }

   Player active_player = env.active_player(*state);
   m_nodes[this_idx].active_player = active_player;
   if(active_player != Player::chance) {
      m_nodes[this_idx].infostate = infostate_map.at(active_player);
   }

   // stage the outgoing edges together with their profile probabilities
   std::vector< Move > moves;
   double prob_sum = 0.;
   size_t feature_base = 0;
   bool is_chance_node = false;
   if constexpr(concepts::stochastic_env< env_type >) {
      if(active_player == Player::chance) {
         is_chance_node = true;
         // every (chance node, outcome) combination gets its own global
         // control-variate feature slot; its exact mean is registered below
         auto outcomes = env.chance_actions(*state);
         feature_base = m_feature_means.size();
         m_feature_means.resize(feature_base + outcomes.size(), 0.);
         moves.reserve(outcomes.size());
         for(size_t i : std::views::iota(size_t{0}, outcomes.size())) {
            const auto& outcome = outcomes[i];
            moves.push_back(Move{
               .move = action_variant_type{outcome},
               .probability = env.chance_probability(*state, outcome),
               .child = 0,
               .feature_id = feature_base + i
            });
            prob_sum += moves.back().probability;
         }
      }
   }
   if(not is_chance_node) {
      auto&& action_policy = profile.at(active_player).at(infostate_map.at(active_player));
      for(const auto& action : env.actions(active_player, *state)) {
         moves.push_back(Move{
            .move = action_variant_type{action},
            .probability = action_policy.at(action),
            .child = 0,
            .feature_id = 0
         });
         prob_sum += moves.back().probability;
      }
   }

   if(prob_sum <= 0. or moves.empty()) {
      throw std::invalid_argument(
         "ProfileGameTree: the given policy profile assigns zero probability mass to all "
         "legal moves of some non-terminal state."
      );
   }
   // defensive normalization against unnormalized input rows
   for(auto& move : moves) {
      move.probability /= prob_sum;
   }
   if(is_chance_node) {
      // register the exact chance-event indicator means AFTER normalization so
      // that they describe precisely the distribution the sampler draws from
      for(auto move_idx : std::views::iota(size_t{0}, moves.size())) {
         m_feature_means[moves[move_idx].feature_id] += reach_probability
                                                        * moves[move_idx].probability;
      }
   }

   // recurse into every child (the imaginary ones included -- AIVAT's
   // correction terms need values of ALL successors of a decision point)
   for(auto move_idx : std::views::iota(size_t{0}, moves.size())) {
      const auto& move = moves[move_idx];
      uptr< world_state_type > next_state_uptr;
      decltype(observation_buffer) child_observation_buffer{};
      decltype(infostate_map) child_infostate_map{};
      std::visit(
         [&](const auto& concrete_move) {
            next_state_uptr = child_state(env, *state, concrete_move);
            auto [obs_buffer, infostates] = next_infostate_and_obs_buffers(
               env, observation_buffer, infostate_map, *state, concrete_move, *next_state_uptr
            );
            child_observation_buffer = std::move(obs_buffer);
            child_infostate_map = std::move(infostates);
         },
         move.move
      );
      moves[move_idx].child = _build(
         env,
         std::move(next_state_uptr),
         reach_probability * move.probability,
         std::move(child_observation_buffer),
         std::move(child_infostate_map),
         profile
      );
   }

   // commit the staged outgoing edges (children are materialized by now)
   m_nodes[this_idx].moves = std::move(moves);

   // backward pass: exact expected values of the profile from this node on.
   // The summation order matches the one AIVAT's correction terms will use,
   // so that exact successor values make the corrections telescope exactly.
   _accumulate_values(this_idx);

   return this_idx;
}

template < typename Env, typename Policy, typename ActionPolicy >
   requires concepts::fosg< std::remove_cvref_t< Env > > and (concepts::state_policy_no_default< Policy, auto_info_state_type< std::remove_cvref_t< Env > >, auto_action_type< std::remove_cvref_t< Env > >, ActionPolicy > or concepts::state_policy_view< Policy, auto_info_state_type< std::remove_cvref_t< Env > >, auto_action_type< std::remove_cvref_t< Env > > >)
void ProfileGameTree< Env, Policy, ActionPolicy >::_accumulate_values(size_t node_idx)
{
   Node& node = m_nodes[node_idx];
   node.values = rm::PlayerValueTable{};
   for(auto [pidx, player] : std::views::enumerate(m_players)) {
      (void) pidx;
      double value = 0.;
      for(const auto& move : node.moves) {
         value += move.probability * m_nodes[move.child].values.at(player);
      }
      node.values.emplace(player, value);
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////// sampler /////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

template < typename Env, typename Policy >
size_t PlayoutSampler< Env, Policy >::_draw_move(const typename tree_type::Node& node)
{
   const double u = m_uniform_01(m_rng);
   double acc = 0.;
   for(size_t i : std::views::iota(size_t{0}, node.moves.size())) {
      acc += node.moves[i].probability;
      if(u < acc) {
         return i;
      }
   }
   return node.moves.size() - 1;
}

template < typename Env, typename Policy >
Playout PlayoutSampler< Env, Policy >::_sample_one()
{
   Playout out;
   size_t node_idx = 0;
   while(not m_tree.node(node_idx).is_terminal) {
      const auto& node = m_tree.node(node_idx);
      const size_t move_idx = _draw_move(node);
      out.path.emplace_back(static_cast< uint32_t >(node_idx), static_cast< uint32_t >(move_idx));
      node_idx = node.moves[move_idx].child;
   }
   out.terminal_node = static_cast< uint32_t >(node_idx);
   return out;
}

///////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// MIVAT estimator //////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

template < typename Env, typename Policy >
void MivatEstimator< Env, Policy >::run(size_t n_playouts)
{
   reset();
   auto playouts = m_sampler.sample(n_playouts);

   const size_t n_total = playouts.size();
   const size_t n_train = n_total >= 4 ? n_total / 2 : 0;
   const std::span< const Playout > training(playouts.data(), n_train);
   const std::span< const Playout > evaluation(playouts.data() + n_train, n_total - n_train);

   const size_t n_features = m_tree.feature_count();
   const auto& feature_means = m_tree.feature_means();

   // ---- training split: least squares fit of the observed rewards onto the
   // centered chance-event indicator features (paper's variance-minimizing
   // closed form solution)

   struct TrainingSample {
      /// sparse centered feature vector: (feature id, x - E[x])
      std::vector< std::pair< size_t, double > > centered_features;
      /// raw rewards indexed like tree().players()
      std::vector< double > rewards;
   };
   std::vector< detail::OnlineStats > train_stats(m_tree.players().size());
   std::vector< TrainingSample > samples;
   samples.reserve(training.size());
   for(const auto& playout : training) {
      TrainingSample sample{};
      for(const auto& [node_idx, move_idx] : playout.path) {
         const auto& node = m_tree.node(node_idx);
         if(node.active_player != Player::chance) {
            continue;
         }
         const auto& move = node.moves[move_idx];
         sample.centered_features.emplace_back(
            move.feature_id, 1. - feature_means[move.feature_id]
         );
      }
      sample.rewards.resize(m_tree.players().size());
      const auto& terminal = m_tree.node(playout.terminal_node);
      for(auto [pidx, player] : std::views::enumerate(m_tree.players())) {
         sample.rewards[pidx] = terminal.rewards.at(player);
         train_stats[pidx].push(sample.rewards[pidx]);
      }
      samples.push_back(std::move(sample));
   }

   // coefficients beta[player][feature]; zeros make the estimator degrade
   // gracefully to plain raw sampling (which stays unbiased!)
   std::vector< std::vector< double > > betas(m_tree.players().size());
   for(auto& beta : betas) {
      beta.assign(n_features, 0.);
   }
   if(n_features > 0 and not samples.empty()) {
      // streaming normal equations of the centered data:
      //     (X^T X) beta = X^T yc   with yc = y - mean(y_train)
      std::vector< double > gramian(n_features * n_features, 0.);
      std::vector< double > rhs(m_tree.players().size() * n_features, 0.);
      double max_diag = 0.;
      for(const auto& sample : samples) {
         for(const auto& [fid_a, xa] : sample.centered_features) {
            for(const auto& [fid_b, xb] : sample.centered_features) {
               gramian[fid_a * n_features + fid_b] += xa * xb;
            }
         }
         for(size_t pidx : std::views::iota(size_t{0}, m_tree.players().size())) {
            const double yc = sample.rewards[pidx] - train_stats[pidx].mean();  // centered target
            for(const auto& [fid, xf] : sample.centered_features) {
               rhs[pidx * n_features + fid] += xf * yc;
            }
         }
      }
      for(size_t f : std::views::iota(size_t{0}, n_features)) {
         max_diag = std::max(max_diag, gramian[f * n_features + f]);
      }
      // tiny Tikhonov term purely for numerical stability on rank-deficient
      // systems (mutually exclusive one-hot blocks are linearly dependent):
      // ANY coefficient vector keeps the estimator unbiased
      const double ridge = std::max(1e-12, 1e-9 * max_diag);
      for(size_t f : std::views::iota(size_t{0}, n_features)) {
         gramian[f * n_features + f] += ridge;
      }
      for(size_t pidx : std::views::iota(size_t{0}, m_tree.players().size())) {
         betas[pidx] = detail::solve_linear_system(
            gramian,
            std::vector< double >{
               rhs.begin() + static_cast< long >(pidx * n_features),
               rhs.begin() + static_cast< long >((pidx + 1) * n_features)
            }
         );
      }
   }

   // ---- evaluation split: frozen coefficients applied to held-out playouts.
   // Because beta is independent of this randomness and E[x] is known exactly
   // from the full-tree enumeration,
   //     estimate = mean(u_eval) - beta . (mean(x_eval) - E[x])
   // is unbiased regardless of the regression quality.
   std::vector< detail::OnlineStats > raw_stats(m_tree.players().size());
   std::vector< detail::OnlineStats > reduced_stats(m_tree.players().size());
   m_playout_estimates.assign(m_tree.players().size(), {});
   if(evaluation.empty()) {
      return;
   }
   std::vector< double > feature_sums(n_features, 0.);
   for(const auto& playout : evaluation) {
      const auto& terminal = m_tree.node(playout.terminal_node);
      // gather the sparse centered feature vector of this playout once
      std::vector< std::pair< size_t, double > > centered_features;
      for(const auto& [node_idx, move_idx] : playout.path) {
         const auto& node = m_tree.node(node_idx);
         if(node.active_player != Player::chance) {
            continue;
         }
         const auto& move = node.moves[move_idx];
         feature_sums[move.feature_id] += 1.;
         centered_features.emplace_back(move.feature_id, 1. - feature_means[move.feature_id]);
      }
      for(auto [pidx, player] : std::views::enumerate(m_tree.players())) {
         const double u = terminal.rewards.at(player);
         double correction = 0.;
         for(const auto& [fid, xc] : centered_features) {
            correction += betas[pidx][fid] * xc;
         }
         const double residual = u - correction;
         raw_stats[pidx].push(u);
         reduced_stats[pidx].push(residual);
         m_playout_estimates[pidx].push_back(residual);
      }
   }
   const double n_eval = static_cast< double >(evaluation.size());
   for(auto [pidx, player] : std::views::enumerate(m_tree.players())) {
      PlayerEvaluation result{};
      result.raw_mean = raw_stats[pidx].mean();
      result.estimate = result.raw_mean;
      for(size_t f : std::views::iota(size_t{0}, n_features)) {
         result.estimate -= betas[pidx][f] * (feature_sums[f] / n_eval - feature_means[f]);
      }
      result.variance.raw_variance = raw_stats[pidx].variance();
      result.variance.reduced_variance = reduced_stats[pidx].variance();
      m_results.emplace(player, result);
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// AIVAT estimator //////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

template < typename Env, typename Policy >
void AivatEstimator< Env, Policy >::run(size_t n_playouts)
{
   reset();
   std::vector< detail::OnlineStats > raw_stats(m_tree.players().size());
   std::vector< detail::OnlineStats > reduced_stats(m_tree.players().size());
   m_playout_estimates.assign(m_tree.players().size(), {});

   auto playouts = m_sampler.sample(n_playouts);
   for(const auto& playout : playouts) {
      const auto& terminal = m_tree.node(playout.terminal_node);
      for(auto [pidx, player] : std::views::enumerate(m_tree.players())) {
         const double u = terminal.rewards.at(player);
         double corrections = 0.;
         for(const auto& [node_idx, move_idx] : playout.path) {
            const auto& node = m_tree.node(node_idx);
            double expectation = 0.;
            for(auto [m, move] : std::views::enumerate(node.moves)) {
               expectation += move.probability * _successor_value(player, node_idx, m);
            }
            // k_h(z) = E_{a~sigma(h,.)}[u_h(a)] - u_h(observed move):
            // conditional mean zero given reaching h (Burch et al., Lemma 1)
            corrections += expectation - _successor_value(player, node_idx, move_idx);
         }
         const double estimate = u + corrections;
         raw_stats[pidx].push(u);
         reduced_stats[pidx].push(estimate);
         m_playout_estimates[pidx].push_back(estimate);
      }
   }

   for(auto [pidx, player] : std::views::enumerate(m_tree.players())) {
      PlayerEvaluation result{};
      result.raw_mean = raw_stats[pidx].mean();
      result.estimate = reduced_stats[pidx].mean();
      result.variance.raw_variance = raw_stats[pidx].variance();
      result.variance.reduced_variance = reduced_stats[pidx].variance();
      m_results.emplace(player, result);
   }
}

#endif  // NOR_EVALUATION_VARIANCE_REDUCTION_TCC

}  // namespace evaluation
}  // namespace nor
