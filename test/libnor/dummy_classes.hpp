
#ifndef NOR_DUMMY_CLASSES_HPP
#define NOR_DUMMY_CLASSES_HPP

#include "nor/fosg_traits.hpp"
#include "nor/game_defs.hpp"
#include "nor/utils/player_informed_type.hpp"

namespace dummy {

struct empty {};

class Action {
  public:
   Action(std::string value, int integer) : m_value(value), m_intvalue(integer) {}

   auto& value() const { return m_value; }

   bool operator==(const Action& other) const { return m_value == other.m_value; }

   int poly_value() const { return 404; }

  private:
   std::string m_value;
   int m_intvalue;
};

struct PolyActionBase {
   virtual ~PolyActionBase() = default;

   virtual int poly_value() const { return m_poly_value; }

   virtual bool operator==(const PolyActionBase& other) const
   {
      return m_poly_value == other.m_poly_value;
   }

   virtual uptr< PolyActionBase > clone() const = 0;

   virtual size_t hash() const { return std::hash< int >{}(m_poly_value); }

  private:
   int m_poly_value = -1;
};

class PolyAction: public PolyActionBase {
  public:
   PolyAction(std::string value, int integer)
       : PolyActionBase(), m_value(value), m_intvalue(integer)
   {
   }

   auto& value() const { return m_value; }

   uptr< PolyActionBase > clone() const override { return std::make_unique< PolyAction >(*this); };

   bool operator==(const PolyActionBase& other) const override
   {
      if(typeid(other) == typeid(*this)) {
         const auto& other_cast = static_cast< const PolyAction& >(other);
         return m_value == other_cast.m_value and m_intvalue == other_cast.m_intvalue;
      } else
         return false;
   }

   int poly_value() const override { return m_intvalue; }

   size_t hash() const override { return std::hash< std::string >{}(m_value); }

  private:
   std::string m_value;
   int m_intvalue;
};

class ChanceOutcome {
  public:
   ChanceOutcome(int value = 0) : m_value(value) {}

   auto& value() const { return m_value; }
   bool operator==(const ChanceOutcome& other) const { return m_value == other.m_value; }

  private:
   int m_value;
};

class Publicstate {
  public:
   using action_type = int;
   using observation_type = std::string;

   size_t size() const;
   bool operator==(const Publicstate&) const;

   void update(observation_type);

   observation_type& operator[](size_t);
};

class Infostate {
  public:
   using action_type = int;
   using observation_type = std::string;

   nor::Player player() const;

   size_t size() const;
   std::pair<
      nor::ObservationHolder< observation_type >,
      nor::ObservationHolder< observation_type > >&
   latest() const;

   bool operator==(const Infostate&) const;

   void update(observation_type, observation_type);

   std::pair<
      nor::ObservationHolder< observation_type >,
      nor::ObservationHolder< observation_type > >&
   operator[](size_t);
};
}  // namespace dummy
namespace std {
template <>
struct hash< dummy::Action > {
   bool operator()(const dummy::Action&) const;
};
template <>
struct hash< dummy::PolyActionBase > {
   size_t operator()(const dummy::PolyActionBase& action) const { return action.hash(); }
};
template <>
struct hash< dummy::ChanceOutcome > {
   bool operator()(const dummy::ChanceOutcome&) const;
};
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

template < bool deterministic = true >
class Env {
  public:
   // nor fosg typedefs
   using world_state_type = struct State {};
   using info_state_type = Infostate;
   using public_state_type = Publicstate;
   using action_type = Action;
   using chance_outcome_type = std::conditional_t< deterministic, void, ChanceOutcome >;
   using observation_type = std::string;
   using action_variant_type = nor::
      action_variant_type_generator_t< action_type, chance_outcome_type >;
   // nor fosg traits
   static constexpr size_t max_player_count() { return 10; }
   static constexpr size_t player_count() { return 10; }
   static constexpr bool serialized() { return false; }
   static constexpr bool unrolled() { return false; }
   static constexpr nor::Stochasticity stochasticity()
   {
      if constexpr(deterministic)
         return nor::Stochasticity::deterministic;
      else
         return nor::Stochasticity::choice;
   }

   std::vector< nor::ActionHolder< action_type > >
   actions(nor::Player player, const world_state_type& wstate) const;
   std::vector< nor::ActionHolder< action_type > > actions(const info_state_type& istate) const;
   std::vector< nor::PlayerInformedType< std::optional< action_variant_type > > >
   private_history(nor::Player, const world_state_type& wstate) const;
   std::vector< nor::PlayerInformedType< std::optional< action_variant_type > > >
   public_history(nor::Player, const world_state_type& wstate) const;
   std::vector< nor::PlayerInformedType< action_variant_type > > open_history(
      const world_state_type& wstate
   ) const;
   std::vector< nor::Player > players(const world_state_type& wstate);
   bool is_partaking(const world_state_type&, nor::Player);
   nor::Player active_player(const world_state_type& wstate) const;
   void reset(world_state_type& wstate);
   bool is_terminal(const world_state_type& wstate);
   double reward(nor::Player player, world_state_type& wstate) const;
   void transition(world_state_type& worldstate, const action_type& action);
   observation_type
   private_observation(nor::Player, const world_state_type&, const action_type&, const world_state_type&);
   observation_type
   public_observation(const world_state_type&, const action_type&, const world_state_type&);
};

struct Traits {
   using action_type = int;
   using info_state_type = std::string;
};

struct TraitsSuperClass {
   using action_type = int;
   using info_state_type = std::string;

   using world_state_type = std::size_t;
   using public_state_type = uint8_t;
};

}  // namespace dummy

namespace nor {
template <>
struct fosg_traits< dummy::Traits > {
   using action_type = int;
   using info_state_type = std::string;
   using world_state_type = std::size_t;
};

}  // namespace nor

#endif  // NOR_DUMMY_CLASSES_HPP
