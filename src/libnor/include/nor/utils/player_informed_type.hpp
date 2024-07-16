
#ifndef NOR_PLAYER_INFORMED_TYPE_HPP
#define NOR_PLAYER_INFORMED_TYPE_HPP

#include "common/common.hpp"
#include "nor/game_defs.hpp"
#include "nor/type_defs.hpp"
#include "nor/utils/utils.hpp"

namespace nor {

/**
 * Simple wrapper class to add the corresponding player as a (non-modifiable) member to the type.
 * @tparam Contained the type to augment with player information
 */
template < typename Contained >
class PlayerInformedType {
  public:
   PlayerInformedType(Contained value, Player p) : m_value(std::move(value)), m_player(p) {}

   /// implicit conversion operator to handle the contained value as if it had never been augmented
   /// by player information
   operator Contained&() { return m_value; }
   /// get the contained value by reference to modify
   auto& value() { return m_value; }
   /// const accessors of the contained value
   [[nodiscard]] auto& value() const { return m_value; }
   /// access to the corresponding player (non-modifiable)
   [[nodiscard]] Player player() const { return m_player; }

   auto to_string() const
      requires(requires(Contained c) { c.to_string(); } or common::printable_v< Contained >)
   {
      return fmt::format("{}\n{}", m_player, m_value);
   };

  private:
   Contained m_value;
   Player m_player;
};

}  // namespace nor

namespace common {

template < typename T >
   requires printable_v< T >
struct printable< nor::PlayerInformedType< T > >: std::true_type {};

template < typename T >
   requires printable_v< T >
std::string to_string(const nor::PlayerInformedType< T >& pi_type)
{
   return pi_type.to_string();
}

}  // namespace common

#endif  // NOR_PLAYER_INFORMED_TYPE_HPP
