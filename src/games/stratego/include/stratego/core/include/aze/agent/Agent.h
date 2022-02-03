#pragma once

#include <random>
#include <range/v3/all.hpp>
#include <vector>

#include "aze/game/Defs.h"

namespace aze {

template < class StateType >
class Agent {
  public:
   using state_type = StateType;
   using action_type = typename state_type::action_type;

   explicit Agent(Team team) : m_team(team) {}
   Agent(const Agent &) = default;
   Agent(Agent &&) noexcept = default;
   Agent &operator=(const Agent &) = default;
   Agent &operator=(Agent &&) noexcept = default;
   virtual ~Agent() = default;

   virtual action_type decide_action(
      const state_type &state,
      const std::vector< action_type > &poss_moves) = 0;

   [[nodiscard]] auto team() const { return m_team; }

  private:
   Team m_team;
};

template < class StateType >
class RandomAgent: public Agent< StateType > {
   using base_type = Agent< StateType >;
   using base_type::base_type;
   using action_type = typename base_type::action_type;

  public:
   explicit RandomAgent(Team team, unsigned int seed = std::random_device{}())
       : base_type(team), mt{seed}
   {
   }

   action_type decide_action(
      const StateType & /*state*/,
      const std::vector< action_type > &poss_moves) override
   {
      std::array< action_type, 1 > selected_move{action_type{{0, 0}, {0, 0}}};
      std::sample(poss_moves.begin(), poss_moves.end(), selected_move.begin(), 1, mt);

      return selected_move[0];
   }

  private:
   std::mt19937 mt;
};

template < class StateType >
class FixedAgent: public Agent< StateType > {
   using base_type = Agent< StateType >;
   using base_type::base_type;
   using action_type = typename base_type::action_type;

  public:
   explicit FixedAgent(Team team, std::vector< action_type > actions)
       : base_type(team), m_actions(actions)
   {
   }

   action_type decide_action(
      const StateType & /*unused*/,
      const std::vector< action_type > &poss_actions) override
   {
      auto action = m_actions.back();
      m_actions.pop_back();

      if(ranges::find_first_of(action, poss_actions) == action.end()) {
         throw std::logic_error(
            "Latest action of scripted actor not in agreement with current possible actions to "
            "choose from.");
      }
      return action;
   }

  private:
   std::vector< action_type > m_actions;
};

template < class StateType >
class InputAgent: public Agent< StateType > {
   using base_type = Agent< StateType >;
   using base_type::base_type;
   using action_type = typename base_type::action_type;

  public:
   explicit InputAgent(
      Team team,
      std::function< std::string(const StateType &state) > repr = nullptr)
       : base_type(team),
         m_repr(
            repr == nullptr ? [&](const auto &state) { return state.to_string(team, true); }
                            : std::move(repr))
   {
   }

   action_type decide_action(const StateType &state, const std::vector< action_type > &poss_actions)
      override
   {
      std::cout << "Current game state:\n";
      std::cout << m_repr(state) << "\n";
      std::cout << "Please choose action integer from possible actions:\n";
      for(const auto &[n, action] : ranges::views::enumerate(poss_actions)) {
         std::cout << n << ": " << action << "\n";
      }
      int choice{};
      std::cin >> choice;
      return poss_actions[choice];
   }

  private:
   std::function< std::string(const StateType &state) > m_repr;
};

}  // namespace aze