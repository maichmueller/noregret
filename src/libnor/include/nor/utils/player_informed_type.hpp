
#ifndef NOR_PLAYER_INFORMED_TYPE_HPP
#define NOR_PLAYER_INFORMED_TYPE_HPP

#include "common/common.hpp"
#include "nor/type_defs.hpp"

namespace nor {

template < typename Contained >
class PlayerInformedType {
  public:
   PlayerInformedType(const Contained& value, Player p) : m_value(value), m_player(p) {}
   PlayerInformedType(Contained&& value, Player p) : m_value(std::move(value)), m_player(p) {}

   /// implicit conversion operator to handle the contained value as if it had never been augmented
   /// by player information
   operator Contained&() { return m_value; }
   /// get the contained value by reference to modify
   auto& value() { return m_value; }
   /// const accessors of the contained values
   [[nodiscard]] auto& value() const { return m_value; }
   [[nodiscard]] Player player() const { return m_player; }

   auto to_string() const
      requires(requires(Contained c) { c.to_string(); } or common::printable_v< Contained >)
   {
      if constexpr(common::printable_v< Contained >) {
         return common::to_string(m_player) + "\n" + common::to_string(m_value);
      } else {
         return common::to_string(m_player) + "\n" << m_value.to_string();
      }
   };

  private:
   Contained m_value;
   Player m_player;
};

}

namespace common {

template < typename T >
   requires printable_v< T >
struct printable< nor::PlayerInformedType< T > >: std::true_type {
};

template < typename T >
   requires printable_v< T >
std::string to_string(const nor::PlayerInformedType< T >& pi_type)
{
   return pi_type.to_string();
}

}  // namespace common

#endif  // NOR_PLAYER_INFORMED_TYPE_HPP
