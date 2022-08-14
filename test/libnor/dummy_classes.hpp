
#ifndef NOR_DUMMY_CLASSES_HPP
#define NOR_DUMMY_CLASSES_HPP

#include "nor/fosg_traits.hpp"
#include "nor/game_defs.hpp"

namespace dummy {

class Publicstate {
  public:
   using action_type = int;
   using observation_type = std::string;

   size_t size() const;
   bool operator==(const Publicstate&) const;

   observation_type& append(observation_type);

   observation_type& operator[](size_t);
};

class Infostate: public Publicstate {
  public:
   nor::Player player() const;
};
}  // namespace dummy
namespace std {
template <>
struct hash< dummy::Infostate > {
   bool operator()(const dummy::Infostate&) const;
};
template <>
struct hash< dummy::Publicstate > {
   bool operator()(const dummy::Publicstate&) const;
};
}  // namespace std

namespace dummy {
class Env {
  public:
   // nor fosg typedefs
   using world_state_type = struct State {
   };
   using info_state_type = Infostate;
   using public_state_type = Publicstate;
   using action_type = int;
   using observation_type = std::string;
   // nor fosg traits
   static constexpr size_t max_player_count() { return 10; }
   static constexpr size_t player_count() { return 10; }
   static constexpr nor::TurnDynamic turn_dynamic() { return nor::TurnDynamic::simultaneous; }
   static constexpr nor::Stochasticity stochasticity() { return nor::Stochasticity::deterministic; }

   std::vector< action_type > actions(nor::Player player, const world_state_type& wstate) const;
   std::vector< action_type > actions(const info_state_type& istate) const;
   std::vector<
      nor::PlayerInformedType< std::optional< std::variant< std::monostate, action_type > > > >
   private_history(nor::Player, const world_state_type& wstate) const;
   std::vector<
      nor::PlayerInformedType< std::optional< std::variant< std::monostate, action_type > > > >
   public_history(nor::Player, const world_state_type& wstate) const;
   std::vector< nor::PlayerInformedType< std::variant< std::monostate, action_type > > >
   open_history(const world_state_type& wstate) const;
   std::vector< nor::Player > players(const world_state_type& wstate);
   bool is_partaking(const world_state_type&, nor::Player);
   nor::Player active_player(const world_state_type& wstate) const;
   void reset(world_state_type& wstate);
   bool is_terminal(world_state_type& wstate);
   double reward(nor::Player player, world_state_type& wstate) const;
   void transition(world_state_type& worldstate, const action_type& action);
   observation_type private_observation(nor::Player player, const world_state_type& wstate);
   observation_type private_observation(nor::Player player, const action_type& action);
   observation_type public_observation(const world_state_type& wstate);
   observation_type public_observation(const action_type& action);
};

struct Traits {
   using action_type = int;
   using info_state_type = double;
};

struct TraitsSuperClass {
   using action_type = int;
   using info_state_type = double;

   using world_state_type = std::string;
   using public_state_type = double;
};

}  // namespace dummy

namespace nor {
template <>
struct fosg_traits< dummy::Traits > {
   using action_type = double;
   using info_state_type = size_t;
   using world_state_type = std::string;
};

}  // namespace nor
#endif  // NOR_DUMMY_CLASSES_HPP
