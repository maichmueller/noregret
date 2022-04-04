#include <gtest/gtest.h>

#include "../games/stratego/fixtures.hpp"
#include "nor/nor.hpp"
#include "nor/wrappers.hpp"

using namespace nor;

TEST(KuhnPoker, vanilla_cfr_usage_kuhnpoker)
{
   //      auto env = std::make_shared< Environment >(std::make_unique< Logic >());
   games::kuhn::Environment env{};
   UniformPolicy uniform_policy = rm::factory::
      make_uniform_policy< games::kuhn::InfoState, HashmapActionPolicy< games::kuhn::Action > >();

   auto tabular_policy = rm::factory::make_tabular_policy(
      std::unordered_map< games::kuhn::InfoState, HashmapActionPolicy< games::kuhn::Action > >{});

   constexpr rm::CFRConfig cfr_config{.alternating_updates = false, .store_public_states = false};

   auto cfr_runner = rm::factory::make_vanilla< cfr_config, true >(
      std::move(env),
      std::make_unique< games::kuhn::State >(),
      tabular_policy,
      std::move(uniform_policy));
   cfr_runner.initialize();
   auto normalize_infostate_policy = [](const auto& policy_table) {
      std::remove_cvref_t< decltype(policy_table) > normalized_table;
      for(const auto& [infostate, policy] : policy_table) {
         normalized_table[infostate] = policy;
         double sum = 0.;
         for(auto [action, value] : policy) {
            sum += value;
         }
         for(auto [action, value] : policy) {
            normalized_table[infostate][action] = value / sum;
         }
      }
      return normalized_table;
   };
   auto prev_policy_profile = normalize_infostate_policy(
      cfr_runner.average_policy().at(Player::alex).table());

   for(int i = 0; i < 20; i++) {
      cfr_runner.iterate(1);

      auto curr_policy_profile = normalize_infostate_policy(
         cfr_runner.average_policy().at(Player::alex).table());

      double total_dev = 0.;
      for(const auto& [curr_pol, prev_pol] :
          ranges::views::zip(curr_policy_profile, prev_policy_profile)) {
         const auto& [curr_istate, curr_istate_pol] = curr_pol;
         const auto& [prev_istate, prev_istate_pol] = prev_pol;
         for(const auto& [curr_pol_state_pol, prev_pol_state_pol] :
             ranges::views::zip(curr_istate_pol, prev_istate_pol)) {
            total_dev = std::abs(
               std::get< 1 >(curr_pol_state_pol) - std::get< 1 >(prev_pol_state_pol));
         }
      }
      prev_policy_profile = std::move(curr_policy_profile);
      std::cout << "iteration: " << i << " - total deviation: " << total_dev << "\n";
   }
   //   std::cout << curr_policy.at(Player::alex).table().size();
   //   LOGD2("Table size", curr_policy.at(Player::alex).table().size());
}
//
//TEST_F(StrategoState3x3, vanilla_cfr_usage_stratego)
//{
//   //      auto env = std::make_shared< Environment >(std::make_unique< Logic >());
//   games::stratego::Environment env{std::make_unique< games::stratego::Logic >()};
//   UniformPolicy uniform_policy = rm::factory::make_uniform_policy<
//      games::stratego::InfoState,
//      HashmapActionPolicy< games::stratego::Action > >();
//
//   auto tabular_policy = rm::factory::make_tabular_policy(
//      std::unordered_map<
//         games::stratego::InfoState,
//         HashmapActionPolicy< games::stratego::Action > >{});
//
//   constexpr rm::CFRConfig cfr_config{.alternating_updates = false, .store_public_states = false};
//
//   auto cfr_runner = rm::factory::make_vanilla< cfr_config, true >(
//      std::move(env),
//      std::make_unique< State >(std::move(state)),
//      tabular_policy,
//      std::move(uniform_policy));
//   cfr_runner.initialize();
//   const auto& curr_policy = *cfr_runner.iterate(100);
//   std::cout << curr_policy.at(Player::alex).table().size();
//   LOGD2("Table size", curr_policy.at(Player::alex).table().size());
//}
