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
      std::vector< action_type > selected_move;
      selected_move.reserve(1);
      std::sample(poss_moves.begin(), poss_moves.end(), std::back_inserter(selected_move), 1, mt);

      return selected_move[0];
   }

  private:
   utils::random::RNG mt;
};

template < class StateType >
class FixedAgent: public Agent< StateType > {
   using base_type = Agent< StateType >;
   using base_type::base_type;
   using action_type = typename base_type::action_type;

  public:
   FixedAgent(Team team, std::vector< action_type > actions) : base_type(team), m_actions()
   {
      m_actions.reserve(actions.size());
      std::copy(actions.rbegin(), actions.rend(), std::back_inserter(m_actions));
   }
   FixedAgent(Team team, std::vector< typename action_type::move_type > moves)
       : base_type(team), m_actions()
   {
      m_actions.reserve(moves.size());
      std::transform(
         moves.rbegin(),
         moves.rend(),
         std::back_inserter(m_actions),
         [&](typename action_type::move_type &move) {
            return action_type{team, std::move(move)};
         });
   }

   action_type decide_action(
      const StateType & /*unused*/,
      const std::vector< action_type > &poss_actions) override
   {
      auto action = m_actions.back();
      m_actions.pop_back();

      if(ranges::find(poss_actions, action) == poss_actions.end()) {
         std::stringstream ss;
         ss << "Latest action of scripted actor not in agreement with current possible actions to "
               "choose from.\n";
         ss << "Action: " << action << "\n";
         ss << "Possible actions: " << aze::utils::VectorPrinter{poss_actions} << "\n";
         throw std::logic_error(ss.str());
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