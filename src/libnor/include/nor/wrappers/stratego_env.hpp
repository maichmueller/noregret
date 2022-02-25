
#ifndef NOR_STRATEGO_ENV_HPP
#define NOR_STRATEGO_ENV_HPP

#include <range/v3/all.hpp>
#include <string>
#include <vector>

#include "nor/game_defs.hpp"
#include "stratego/Game.hpp"

namespace nor::games {

using stratego_observation = std::string;

class StrategoInfostate {
  public:
   using action_type = stratego::Action;
   using observation_type = stratego_observation;

   StrategoInfostate() = default;

   [[nodiscard]] auto& history() const { return m_history; }

   template < typename... Args >
   auto append(Args&&... args)
   {
      auto ret_val = m_history.emplace_back(std::forward< Args >(args)...);
      _hash();
      return ret_val;
   }

   [[nodiscard]] size_t hash() const { return m_hash_cache; }

   bool operator==(const StrategoInfostate& other) const
   {
      auto zip_view = ranges::views::zip(m_history, other.history());
      return std::all_of(zip_view.begin(), zip_view.end(), [](const auto& tuple) {
         const auto& [this_hist_elem, other_hist_elem] = tuple;
         return this_hist_elem.first == other_hist_elem.first
                && this_hist_elem.second == other_hist_elem.second;
      });
   }
   inline bool operator!=(const StrategoInfostate& other) const { return not (*this == other); }

  private:
   std::vector< std::pair< stratego::Action, std::string > > m_history;
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
      size_t hash = std::hash< stratego_observation >{}(ss.str());
      m_hash_cache = hash;
      return hash;
   }
};

using StrategoPublicstate = StrategoInfostate;

}  // namespace nor::games

namespace std {
template <>
struct hash< nor::games::StrategoInfostate > {
   size_t operator()(const nor::games::StrategoInfostate& state) const noexcept
   {
      return state.hash();
   }
};
}  // namespace std

namespace nor::games {

class NORStrategoEnv {
   using Team = stratego::Team;
   using world_state_type = stratego::State;
   using info_state_type = StrategoInfostate;
   using public_state_type = StrategoPublicstate;
   static constexpr size_t max_player_count = 2;
   static constexpr TurnDynamic turn_dynamic = TurnDynamic::sequential;

  public:
   explicit NORStrategoEnv(uptr< stratego::Logic >&& logic) : m_logic(std::move(logic)) {}

   std::vector< stratego::Action > actions(Player player, world_state_type& wstate);
   static inline std::vector< Player > players() { return {Player::alex, Player::bob}; }
   auto reset(world_state_type& wstate);
   bool is_terminal(world_state_type& wstate);
   static double reward(Player player, world_state_type& wstate);

   void transition(const stratego::Action& action, world_state_type& worldstate);

   static inline auto to_team(const Player& player)
   {
      {
         return stratego::Team(static_cast< size_t >(player));
      }
   }
   static inline auto to_player(const Team& team)
   {
      {
         return Player(static_cast< size_t >(team));
      }
   }

  private:
   uptr< stratego::Logic > m_logic;
   static double _status_to_reward(stratego::Status status, Player player);
};

}  // namespace nor::games

#endif  // NOR_STRATEGO_ENV_HPP
