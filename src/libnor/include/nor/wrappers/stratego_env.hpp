
#ifndef NOR_STRATEGO_ENV_HPP
#define NOR_STRATEGO_ENV_HPP

#include <range/v3/all.hpp>
#include <string>
#include <vector>

#include "nor/game_defs.hpp"
#include "stratego/Game.hpp"

namespace nor::games::stratego {

using namespace ::stratego;

class InfoState {
  public:
   using world_state_type = State;
   using observation_type = std::string;

   InfoState() = default;

   [[nodiscard]] auto& history() const { return m_history; }
   [[nodiscard]] size_t size() const { return m_history.size(); }

   template < typename... Args >
   auto& append(Args&&... args)
   {
      auto& ret_val = m_history.emplace_back(std::forward< Args >(args)...);
      _hash();
      return ret_val;
   }

   [[nodiscard]] size_t hash() const { return m_hash_cache; }
   [[nodiscard]] Player player() const { return m_player; }

   bool operator==(const InfoState& other) const
   {
      auto zip_view = ranges::views::zip(m_history, other.history());
      return std::all_of(zip_view.begin(), zip_view.end(), [](const auto& tuple) {
         const auto& [this_hist_elem, other_hist_elem] = tuple;
         return this_hist_elem.first == other_hist_elem.first
                && this_hist_elem.second == other_hist_elem.second;
      });
   }
   inline bool operator!=(const InfoState& other) const { return not (*this == other); }

  private:
   std::vector< std::pair< observation_type, observation_type > > m_history;
   Player m_player;
   size_t m_hash_cache{0};

   size_t _hash()
   {
      std::stringstream ss;
      size_t pos = 0;
      for(const auto& [action, observation] : m_history) {
         ss << "a_" << pos << ":" << action;
         ss << "-";
         ss << "obs_" << pos << ":" << observation;
         pos++;
      }
      size_t hash = std::hash< observation_type >{}(ss.str());
      m_hash_cache = hash;
      return hash;
   }
};

using PublicState = InfoState;

}  // namespace nor::games::stratego

namespace std {
template <>
struct hash< nor::games::stratego::InfoState > {
   size_t operator()(const nor::games::stratego::InfoState& state) const noexcept
   {
      return state.hash();
   }
};
}  // namespace std

namespace nor::games::stratego {

class Environment {
   // nor fosg typedefs
   using world_state_type = State;
   using info_state_type = InfoState;
   using public_state_type = PublicState;
   using action_type = Action;
   using observation_type = InfoState::observation_type;
   // nor fosg traits
   static constexpr size_t max_player_count = 2;
   static constexpr TurnDynamic turn_dynamic = TurnDynamic::sequential;

  public:
   explicit Environment(uptr< Logic >&& logic);

   std::vector< action_type > actions(Player player, const world_state_type& wstate) const;
   std::vector< action_type > actions(const info_state_type& istate) const;
   static inline std::vector< Player > players() const { return {Player::alex, Player::bob}; }
   auto reset(world_state_type& wstate) const;
   static bool is_terminal(world_state_type& wstate);
   static double reward(Player player, world_state_type& wstate);
   void transition(const action_type& action, world_state_type& worldstate) const;
   observation_type private_observation(Player player, const world_state_type& wstate) const;
   observation_type private_observation(Player player, const action_type& action) const;
   observation_type public_observation(Player player, const world_state_type& wstate) const;
   observation_type public_observation(Player player, const action_type& action) const;
   static inline auto to_team(const Player& player)
   {
      {
         return Team(static_cast< size_t >(player));
      }
   }
   static inline auto to_player(const Team& team)
   {
      {
         return Player(static_cast< size_t >(team));
      }
   }

  private:
   uptr< Logic > m_logic;

   static double _status_to_reward(Status status, Player player);
};

}  // namespace nor::games::stratego

#endif  // NOR_STRATEGO_ENV_HPP
