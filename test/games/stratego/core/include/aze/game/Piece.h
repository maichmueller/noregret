
#pragma once

#include <algorithm>
#include <array>
#include <random>

#include "aze/utils/utils.h"


template < typename Position, typename Token >
class Piece {
   /**
    * A typical Piece class holding the most relevant data to describe a piece.
    * Each piece is assigned a team (0 or 1), a PositionType position, and given
    * a Type (int as the basic idea 0-11). Since there can be more than one piece
    * of a type each piece also receives a version (part of the type member).
    * Meta-attributes are 'hidden', 'has_moved'.
    *
    * Null-Pieces are set by the flag 'null_piece', which is necessary
    * since every position on a board needs a piece at any time.
    *
    **/

  public:
   using position_type = Position;
   using token_type = Token;

  protected:
   position_type m_position;
   token_type m_token;
   int m_team;
   bool m_null_piece = false;
   bool m_hidden;
   bool m_has_moved;

  public:
   Piece(position_type pos, token_type type, int team, bool hidden, bool has_moved)
       : m_position(pos), m_token(type), m_team(team), m_hidden(hidden), m_has_moved(has_moved)
   {
   }

   Piece(position_type position, token_type type, int team)
       : Piece(position, type, team, true, false)
   {
   }

   // a Null Piece Constructor
   explicit Piece(const position_type &position)
       : m_position(position),
         m_token(),
         m_team(-1),
         m_null_piece(true),
         m_hidden(false),
         m_has_moved(false)
   {
   }

   // getter and setter methods here only

   void set_flag_has_moved(bool has_moved = true) { this->m_has_moved = has_moved; }

   void set_flag_unhidden(bool h = false) { m_hidden = h; }

   void set_position(position_type p) { m_position = std::move(p); }

   [[nodiscard]] bool is_null() const { return m_null_piece; }

   [[nodiscard]] position_type get_position() const { return m_position; }

   [[nodiscard]] int get_team(bool flip_team = false) const
   {
      return (flip_team) ? 1 - m_team : m_team;
   }

   [[nodiscard]] token_type get_token() const { return m_token; }

   [[nodiscard]] bool get_flag_hidden() const { return m_hidden; }

   [[nodiscard]] bool get_flag_has_moved() const { return m_has_moved; }

   bool operator==(const Piece &other) const
   {
      return other.get_position() == m_position && m_token == other.get_token()
             && m_team == other.get_team() && m_hidden == other.get_flag_hidden()
             && m_has_moved == other.get_flag_has_moved();
   }

   bool operator!=(const Piece &other) const { return ! (*this == other); }
};
