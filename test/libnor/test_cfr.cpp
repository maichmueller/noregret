#include <gtest/gtest.h>

#include <unordered_map>

#include "../games/stratego/fixtures.hpp"
#include "nor/nor.hpp"
#include "nor/wrappers.hpp"

using namespace nor;

TEST(InfostateNode, storage_correctness)
{
   std::unordered_map<
      sptr< nor::games::kuhn::Infostate >,
      rm::InfostateNodeData< nor::games::kuhn::Action >,
      decltype([](const sptr< nor::games::kuhn::Infostate >& ptr) {
         return std::hash< nor::games::kuhn::Infostate >{}(*ptr);
      }),
      decltype([](const sptr< nor::games::kuhn::Infostate >& ptr1,
                  const sptr< nor::games::kuhn::Infostate >& ptr2) { return *ptr1 == *ptr2; }) >
      infonode_map{};

   nor::games::kuhn::Environment env{};
   nor::games::kuhn::State state{};
   auto first_istate_alex = std::make_shared< nor::games::kuhn::Infostate >(nor::Player::alex);
   auto istate_bob = std::make_shared< nor::games::kuhn::Infostate >(nor::Player::bob);
   state.apply_action(nor::games::kuhn::ChanceOutcome{
      nor::games::kuhn::Player::one, nor::games::kuhn::Card::queen});
   state.apply_action(
      nor::games::kuhn::ChanceOutcome{nor::games::kuhn::Player::two, nor::games::kuhn::Card::king});
   first_istate_alex->append(env.private_observation(nor::Player::alex, state));

   auto& first_alex_node_data = infonode_map
                                   .emplace(
                                      first_istate_alex,
                                      rm::InfostateNodeData< nor::games::kuhn::Action >{
                                         env.actions(Player::alex, state)})
                                   .first->second;

   state.apply_action(nor::games::kuhn::Action::check);
   istate_bob->append(env.private_observation(nor::Player::bob, state));
   auto& bob_node_data = infonode_map
                            .emplace(
                               istate_bob,
                               rm::InfostateNodeData< nor::games::kuhn::Action >{
                                  env.actions(Player::bob, state)})
                            .first->second;

   state.apply_action(nor::games::kuhn::Action::bet);
   auto second_istate_alex = std::make_shared< nor::games::kuhn::Infostate >(*first_istate_alex);
   second_istate_alex->append(env.private_observation(nor::Player::alex, state));
   auto& second_alex_node_data = infonode_map
                                    .emplace(
                                       second_istate_alex,
                                       rm::InfostateNodeData< nor::games::kuhn::Action >{
                                          env.actions(Player::alex, state)})
                                    .first->second;
   second_alex_node_data.regret(games::kuhn::Action::check) += 5;
   second_alex_node_data.regret(games::kuhn::Action::bet) -= 10;

   auto& second_alex_node_data_other_ref = infonode_map.at(
      std::make_shared< games::kuhn::Infostate >(*second_istate_alex));

   ASSERT_EQ(
      second_alex_node_data.regret(games::kuhn::Action::check),
      second_alex_node_data_other_ref.regret(games::kuhn::Action::check));
   ASSERT_EQ(
      second_alex_node_data.regret(games::kuhn::Action::bet),
      second_alex_node_data_other_ref.regret(games::kuhn::Action::bet));
}

template < typename Policy >
void print_policy(const Policy& policy)
{
   auto policy_vec = ranges::to_vector(policy | ranges::views::transform([](const auto& kv) {
                                          return std::pair{std::get< 0 >(kv), std::get< 1 >(kv)};
                                       }));
   std::sort(policy_vec.begin(), policy_vec.end(), [](const auto& kv, const auto& other_kv) {
      return std::get< 0 >(kv).history().back().size()
             < std::get< 0 >(other_kv).history().back().size();
   });
   auto action_policy_printer = [&](const auto& action_policy) {
      std::stringstream ss;
      ss << "[ ";
      for(const auto& [key, value] : action_policy) {
         ss << std::setw(5) << common::to_string(key) + ": " << std::setw(6) << std::setprecision(3)
            << value << " ";
      }
      ss << "]";
      return ss.str();
   };
   for(const auto& [istate, action_policy] : policy_vec) {
      std::cout << std::setw(5) << istate.player() << " | " << std::setw(8)
                << common::left(istate.history().back(), 5, " ") << " -> ";
      std::cout << action_policy_printer(action_policy) << "\n";
   }
}

template < typename CFRRunner, typename Policy >
void evaluate_policies(
   Player player,
   CFRRunner& cfr_runner,
   Policy& prev_policy_profile,
   size_t iteration)
{
   auto curr_policy_profile = rm::normalize_state_policy(
      cfr_runner.average_policy().at(player).table());

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
   auto normalized_current_policy = rm::normalize_state_policy(
      cfr_runner.policy().at(player).table());

   std::cout << "Average policy:\n";
   print_policy(rm::normalize_state_policy(cfr_runner.average_policy().at(Player::alex).table()));
   print_policy(rm::normalize_state_policy(cfr_runner.average_policy().at(Player::bob).table()));

   prev_policy_profile = std::move(curr_policy_profile);
   if(cfr_runner.iteration() > 1) {
      auto game_value_map = cfr_runner.game_value();
      std::cout << "iteration: " << iteration << " | game value for player " << player << ": "
                << game_value_map.get()[player] << "\n";
   }
   //   std::cout << "total deviation: " << total_dev << "\n";
}

TEST(RockPaperScissors, vanilla_cfr_usage_rockpaperscissors)
{
   //      auto env = std::make_shared< Environment >(std::make_unique< Logic >());
   games::rps::Environment env{};

   auto avg_tabular_policy = rm::factory::make_tabular_policy(
      std::unordered_map< games::rps::InfoState, HashmapActionPolicy< games::rps::Action > >{},
      rm::factory::
         make_zero_policy< games::rps::InfoState, HashmapActionPolicy< games::rps::Action > >());

   auto tabular_policy_alex = rm::factory::make_tabular_policy(
      std::unordered_map< games::rps::InfoState, HashmapActionPolicy< games::rps::Action > >{},
      rm::factory::
         make_uniform_policy< games::rps::InfoState, HashmapActionPolicy< games::rps::Action > >());

   auto tabular_policy_bob = rm::factory::make_tabular_policy(
      std::unordered_map< games::rps::InfoState, HashmapActionPolicy< games::rps::Action > >{},
      rm::factory::
         make_uniform_policy< games::rps::InfoState, HashmapActionPolicy< games::rps::Action > >());

   auto infostate_alex = games::rps::InfoState{Player::alex};
   auto infostate_bob = games::rps::InfoState{Player::alex};
   auto init_state = games::rps::State();
   infostate_alex.append(env.private_observation(Player::alex, init_state));
   infostate_bob.append(env.private_observation(Player::bob, init_state));
   auto action_alex = games::rps::Action{games::rps::Team::one, games::rps::Hand::rock};

   env.transition(init_state, action_alex);

   infostate_bob.append(env.private_observation(Player::bob, action_alex));
   infostate_bob.append(env.private_observation(Player::bob, init_state));

   // off-set the given policy by very bad initial values to test the algorithm bouncing back
   tabular_policy_alex.emplace(
      infostate_alex,
      HashmapActionPolicy< games::rps::Action >{std::unordered_map{
         std::pair{games::rps::Action{games::rps::Team::one, games::rps::Hand::rock}, 1. / 10.},
         std::pair{games::rps::Action{games::rps::Team::one, games::rps::Hand::paper}, 2. / 10.},
         std::pair{
            games::rps::Action{games::rps::Team::one, games::rps::Hand::scissors}, 7. / 10.}}});

   // off-set the given policy by very bad initial values to test the algorithm bouncing back
   tabular_policy_bob.emplace(
      infostate_bob,
      HashmapActionPolicy< games::rps::Action >{std::unordered_map{
         std::pair{games::rps::Action{games::rps::Team::two, games::rps::Hand::rock}, 5. / 10.},
         std::pair{games::rps::Action{games::rps::Team::two, games::rps::Hand::paper}, 4. / 10.},
         std::pair{
            games::rps::Action{games::rps::Team::two, games::rps::Hand::scissors}, 1. / 10.}}});

   constexpr rm::CFRConfig cfr_config{
      .alternating_updates = false, .store_public_states = false, .store_world_states = true};

   auto cfr_runner = rm::factory::make_vanilla< cfr_config >(
      std::move(env),
      std::make_unique< games::rps::State >(),
      std::unordered_map{
         std::pair{Player::alex, tabular_policy_alex}, std::pair{Player::bob, tabular_policy_bob}},
      std::unordered_map{
         std::pair{Player::alex, avg_tabular_policy}, std::pair{Player::bob, avg_tabular_policy}});

   auto player = Player::alex;

   auto initial_policy_profile = rm::normalize_state_policy(
      cfr_runner.average_policy().at(player).table());

   for(size_t i = 0; i < 20000; i++) {
      cfr_runner.iterate(1);
      evaluate_policies(player, cfr_runner, initial_policy_profile, i);
   }
}

TEST(KuhnPoker, vanilla_cfr_usage_kuhnpoker)
{
   games::kuhn::Environment env{};

   auto avg_tabular_policy = rm::factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{},
      rm::factory::
         make_zero_policy< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >());

   auto tabular_policy = rm::factory::make_tabular_policy(
      std::unordered_map< games::kuhn::Infostate, HashmapActionPolicy< games::kuhn::Action > >{},
      rm::factory::make_uniform_policy<
         games::kuhn::Infostate,
         HashmapActionPolicy< games::kuhn::Action > >());

   constexpr rm::CFRConfig cfr_config{
      .alternating_updates = true, .store_public_states = false, .store_world_states = true};

   auto cfr_runner = rm::factory::make_vanilla< cfr_config, true >(
      std::move(env), std::make_unique< games::kuhn::State >(), tabular_policy, avg_tabular_policy);

   auto player = Player::alex;

   auto initial_policy_profile = rm::normalize_state_policy(
      cfr_runner.average_policy().at(player).table());

   for(size_t i = 0; i < 100; i++) {
      cfr_runner.iterate(1);
      evaluate_policies(player, cfr_runner, initial_policy_profile, i);
   }
}

// TEST_F(StrategoState3x3, vanilla_cfr_usage_stratego)
//{
//    //      auto env = std::make_shared< Environment >(std::make_unique< Logic >());
//    games::stratego::Environment env{std::make_unique< games::stratego::Logic >()};
//    UniformPolicy uniform_policy = rm::factory::make_uniform_policy<
//       games::stratego::InfoState,
//       HashmapActionPolicy< games::stratego::Action > >();
//
//    auto tabular_policy = rm::factory::make_tabular_policy(
//       std::unordered_map<
//          games::stratego::InfoState,
//          HashmapActionPolicy< games::stratego::Action > >{});
//
//    constexpr rm::CFRConfig cfr_config{.alternating_updates = false, .store_public_states =
//    false};
//
//    auto cfr_runner = rm::factory::make_vanilla< cfr_config, true >(
//       std::move(env),
//       std::make_unique< State >(std::move(state)),
//       tabular_policy,
//       std::move(uniform_policy));
//    cfr_runner.initialize();
//    const auto& curr_policy = *cfr_runner.iterate(100);
//    std::cout << curr_policy.at(Player::alex).table().size();
//    LOGD2("Table size", curr_policy.at(Player::alex).table().size());
// }
