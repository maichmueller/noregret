
#include <gtest/gtest.h>

#include "nor/env.hpp"
#include "nor/fosg_helpers.hpp"

using namespace nor;

auto history_string_view(auto& history)
{
   return history | ranges::views::transform([](const auto& av) {
             return std::visit(
                [&](const auto& a) { return common::to_string(common::deref(a)); }, av
             );
          });
}

template < typename Env, typename State >
auto compute_history_to_infostate_map_and_print(Env env, State root, std::string_view env_name)
{
   auto [terminals, istate_imap] = map_histories_to_infostates(env, root, false);
   std::cout << "Environment: " << env_name << "\n";
   std::cout << "History --> Infostate:\n";
   constexpr size_t first_col_width = 20;
   for(auto [history, actives_infostatemap] : istate_imap) {
      const auto& [active_player, infostate_map] = actives_infostatemap;
      std::cout << common::left("Active Player: ", first_col_width, " ")
                << common::to_string(active_player) << "\n"
                << common::left("History: ", first_col_width, " ") << history_string_view(history)
                << "\n"
                << common::left("Infostate: ", first_col_width, " ")
                << (active_player == Player::chance
                       ? std::string{}
                       : infostate_map.at(active_player)
                            ->to_string(common::left("\n", first_col_width + 1, " ")))
                << "\n\n";
   }
   std::cout << "Decision Histories (count: " << istate_imap.size() << "):\n";
   for(auto [history, _] : istate_imap) {
      std::cout << history_string_view(history) << "\n";
   }

   std::cout << "\nTerminal Histories (count: " << terminals.size() << "):\n";
   for(auto terminal_history : terminals) {
      std::cout << history_string_view(terminal_history) << "\n";
   }

   std::cout << std::endl;
   return std::pair{terminals, istate_imap};
}

template < typename Env, typename State >
auto compute_decision_infostates_and_print(Env env, State root, std::string_view env_name)
{
   auto infostates = decision_infostates(env, root);
   std::cout << "Environment: " << env_name << "\n";
   std::cout << "\nDecision Infostates (count: " << infostates.size() << "):\n";
   auto infostate_strings = ranges::to_vector(
      infostates | ranges::view::transform([](const auto& infostate) {
         return std::pair{infostate.player(), infostate.to_string()};
      })
   );
   ranges::sort(infostate_strings);
   for(const auto& player_and_infostate : infostate_strings) {
      const auto& [player, infostate] = player_and_infostate;
      std::cout << "Perspective: " << player << "\tInfostate: " << infostate << "\n";
   }
   std::cout << std::endl;
   return infostates;
}

TEST(HistoryToInfostateMap, rps)
{
   using namespace games::rps;
   auto [terminals, istate_imap] = compute_history_to_infostate_map_and_print(
      Environment{}, State{}, "RPS"
   );
   // we expect 4 infostates, because the first player chooses first at root, then bob chooses at
   // the decision nodes in which player chose Rock, Paper, or Scissors.
   EXPECT_EQ(istate_imap.size(), 4);
   EXPECT_EQ(terminals.size(), 9);
}

TEST(HistoryToInfostateMap, kuhn)
{
   using namespace games::kuhn;
   auto [terminals, istate_imap] = compute_history_to_infostate_map_and_print(
      Environment{}, State{}, "Kuhn-Poker"
   );
   // we expect the istate map to have 28 entries because alex and bob together have 24 decision
   // nodes and chance has 4 (at the root with the empty history, and the after the first card
   // choice jack, queen, or king.)
   EXPECT_EQ(istate_imap.size(), 28);
   // there are 30 terminal nodes in kuhn poker so all should be contained.
   EXPECT_EQ(terminals.size(), 30);
}

TEST(DecisionInfostates, rps)
{
   using namespace games::rps;
   auto infostates = compute_decision_infostates_and_print(Environment{}, State{}, "RPS");
   // we expect 12 infostates in total, 6 for each player
   EXPECT_EQ(infostates.size(), 2);
}

TEST(DecisionInfostates, kuhn)
{
   using namespace games::kuhn;
   auto infostates = compute_decision_infostates_and_print(Environment{}, State{}, "Kuhn-Poker");
   // we expect 12 infostates in total, 6 for each player
   EXPECT_EQ(infostates.size(), 12);
}

TEST(DecisionInfostates, leduc)
{
   using namespace games::leduc;
   auto infostates = compute_decision_infostates_and_print(
      Environment{}, State{LeducConfig{}}, "Leduc-Poker"
   );
   // we expect 12 infostates in total, 6 for each player
   EXPECT_EQ(infostates.size(), 288);
}

TEST(DecisionInfostates, leduc5)
{
   using namespace games::leduc;
   auto infostates = compute_decision_infostates_and_print(
      Environment{}, State{LeducConfig::leduc5()}, "Leduc-Poker"
   );
   // we expect 12 infostates in total, 6 for each player
   EXPECT_EQ(infostates.size(), 34224);
}
