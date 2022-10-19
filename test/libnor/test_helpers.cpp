
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
   auto active_imap = std::get< 1 >(ret);
   auto inactive_imap = std::get< 2 >(ret);
   for(auto [history, infostate] : active_imap) {
      std::cout << ((*history | ranges::views::transform([](const auto& av) {
         return std::visit(
            common::Overload{
               [&](const std::monostate&) { return std::string{}; },
               [&](const auto a) { return common::to_string(a); }},
            av
         );
      }))) << "\n"
                << infostate.to_string() << "\n";
   }
}

TEST(IteratingInformationStates, kuhn_correctness)
{
   auto env = games::kuhn::Environment{};
   auto root = games::kuhn::State{};
   auto ret = map_histories_to_infostates(env, root);
   auto terminals = std::get< 0 >(ret);
   auto active_imap = std::get< 1 >(ret);
   auto inactive_imap = std::get< 2 >(ret);
   for(auto [history, infostate] : active_imap) {
      std::cout << ((*history | ranges::views::transform([](const auto& av) {
         return std::visit([&](const auto a) { return common::to_string(a); }, av);
      }))) << "\n"
                << infostate.to_string() << "\n";
   }
}
