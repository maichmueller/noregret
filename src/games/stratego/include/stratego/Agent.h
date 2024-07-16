#pragma once

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <random>
#include <range/v3/all.hpp>
#include <stdexcept>
#include <vector>

namespace stratego {

template < class StateType >
class Agent {
  public:
   using state_type = StateType;

   explicit Agent(Team team) : m_team(team) {}
   Agent(const Agent &) = default;
   Agent(Agent &&) noexcept = default;
   Agent &operator=(const Agent &) = default;
   Agent &operator=(Agent &&) noexcept = default;
   virtual ~Agent() = default;

   virtual Action
   decide_action(const state_type &state, const std::vector< Action > &poss_moves) = 0;

   [[nodiscard]] auto team() const { return m_team; }

  private:
   Team m_team;
};

template < class StateType >
class RandomAgent: public Agent< StateType > {
   using base_type = Agent< StateType >;
   using base_type::base_type;
   using Action = typename base_type::Action;

  public:
   explicit RandomAgent(Team team, unsigned int seed = std::random_device{}())
       : base_type(team), mt{seed}
   {
   }

   Action decide_action(const StateType & /*state*/, const std::vector< Action > &poss_moves)
      override
   {
      std::vector< Action > selected_move;
      selected_move.reserve(1);
      std::sample(poss_moves.begin(), poss_moves.end(), std::back_inserter(selected_move), 1, mt);

      return selected_move[0];
   }

  private:
   common::RNG mt;
};

template < class StateType >
class FixedAgent: public Agent< StateType > {
   using base_type = Agent< StateType >;
   using base_type::base_type;

  public:
   FixedAgent(Team team, std::vector< Action > actions) : base_type(team), m_actions()
   {
      m_actions.reserve(actions.size());
      std::copy(actions.rbegin(), actions.rend(), std::back_inserter(m_actions));
   }
   FixedAgent(Team team, std::vector< typename Action::move_type > moves)
       : base_type(team), m_actions()
   {
      m_actions.reserve(moves.size());
      std::transform(moves.rbegin(), moves.rend(), std::back_inserter(m_actions), [&](Move &move) {
         return Action{team, std::move(move)};
      });
   }

   Action decide_action(
      const StateType & /*unused*/,
      const std::vector< Action > &available_actions
   ) override
   {
      auto action = m_actions.back();
      m_actions.pop_back();

      if(not ranges::contains(available_actions, action)) {
         throw std::logic_error(fmt::format(
            "Latest action of scripted actor not in agreement with currently available actions to "
            "choose from.\nAction: {}\nPossible actions: {}",
            action,
            available_actions
         ));
      }
      return action;
   }

  private:
   std::vector< Action > m_actions;
};

template < class StateType >
class InputAgent: public Agent< StateType > {
   using base_type = Agent< StateType >;
   using base_type::base_type;
   using Action = typename base_type::Action;

  public:
   explicit InputAgent(
      Team team,
      std::function< std::string(const StateType &state) > repr = nullptr
   )
       : base_type(team),
         m_repr(
            repr == nullptr ? [&](const auto &state) { return state.to_string(team, true); }
                            : std::move(repr)
         )
   {
   }

   Action decide_action(const StateType &state, const std::vector< Action > &available_actions)
      override
   {
      std::cout << "Current game state:\n";
      std::cout << m_repr(state) << "\n";
      std::cout << "Please choose action integer from possible actions:\n";
      for(const auto &[n, action] : ranges::views::enumerate(available_actions)) {
         std::cout << n << ": " << action << "\n";
      }
      int choice{};
      std::cin >> choice;
      return available_actions[choice];
   }

  private:
   std::function< std::string(const StateType &state) > m_repr;
};

}  // namespace stratego
