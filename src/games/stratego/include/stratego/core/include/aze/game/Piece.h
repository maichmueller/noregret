
#pragma once

#include <algorithm>
#include <array>
#include <random>

#include "aze/game/Defs.h"
#include "aze/utils/utils.h"

namespace aze {

template < typename Position, typename Token >
class Piece {
   /**
    * A typical Piece class holding the most relevant data to describe a piece.
    * Each piece is assigned a team, a PositionType position, and given
    * a Type (int as the basic idea 0-11).
    * Meta-attributes are 'hidden', 'has_moved'.
    *
    **/

  public:
   using position_type = Position;
   using token_type = Token;

  protected:
   position_type m_position;
   token_type m_token;
   Team m_team;
   bool m_hidden;
   bool m_has_moved;

  public:
   Piece(position_type pos, token_type type, Team team, bool hidden, bool has_moved)
       : m_position(pos), m_token(type), m_team(team), m_hidden(hidden), m_has_moved(has_moved)
   {
   }

   Piece(position_type position, token_type type, Team team)
       : Piece(position, type, team, true, false)
   {
   }

   Piece(const Piece&) = default;
   Piece(Piece&&) noexcept = default;
   Piece& operator=(const Piece&) = default;
   Piece& operator=(Piece&&) noexcept = default;

   // getter and setter methods here only

   void flag_has_moved(bool has_moved) { this->m_has_moved = has_moved; }

   void flag_unhidden(bool h) { m_hidden = h; }

   void position(position_type p) { m_position = std::move(p); }

   [[nodiscard]] position_type position() const { return m_position; }

   [[nodiscard]] Team team() const { return m_team; }

   [[nodiscard]] token_type token() const { return m_token; }

   [[nodiscard]] bool flag_hidden() const { return m_hidden; }

   [[nodiscard]] bool flag_has_moved() const { return m_has_moved; }

   bool operator==(const Piece& other) const
   {
      return other.position() == m_position && m_token == other.token() && m_team == other.team()
             && m_hidden == other.flag_hidden() && m_has_moved == other.flag_has_moved();
   }

   bool operator!=(const Piece& other) const { return *this != other; }
};
}  // namespace aze