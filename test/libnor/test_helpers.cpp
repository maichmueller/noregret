
#include <gtest/gtest.h>

#include "nor/env.hpp"
#include "nor/fosg_helpers.hpp"

using namespace nor;

TEST(IteratingInformationStates, rps_correctness)
{
   auto env = games::rps::Environment{};
   auto root = games::rps::State{};
   auto ret = map_histories_to_infostates(env, root);
   auto terminals = std::get< 0 >(ret);
   auto istate_imap = std::get< 1 >(ret);
   for(auto [history, actives_infostatemap] : istate_imap) {
      const auto& [active_players, infostate_map] = actives_infostatemap;
      std::cout << ((history | ranges::views::transform([](const auto& av) {
                        return std::visit(
                           [&](const auto& a) { return common::to_string(common::deref(a)); }, av
                        );
                     })))
                << "\n"
                << infostate_map.at(active_players[0])->to_string() << "\n";
   }
}

TEST(IteratingInformationStates, kuhn_correctness)
{
   auto env = games::kuhn::Environment{};
   auto root = games::kuhn::State{};
   auto ret = map_histories_to_infostates(env, root, true);
   auto terminals = std::get< 0 >(ret);
   auto istate_imap = std::get< 1 >(ret);
   for(auto [history, actives_infostatemap] : istate_imap) {
      const auto& [active_players, infostate_map] = actives_infostatemap;
      std::cout << ((history | ranges::views::transform([](const auto& av) {
                        return std::visit(
                           [&](const auto& a) { return common::to_string(a.get()); }, av
                        );
                     })))
                << "\n"
                << (active_players.empty() ? std::string("")
                                           : infostate_map.at(active_players[0])->to_string())
                << "\n";
   }
}
