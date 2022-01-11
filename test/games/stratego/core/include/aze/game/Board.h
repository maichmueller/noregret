
#pragma once

#include <algorithm>
#include <array>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include "Move.h"
#include "Piece.h"
#include "aze/game/Defs.h"
#include "aze/types.h"
#include "aze/utils/logging_macros.h"
#include "aze/utils/utils.h"

template < typename PieceType >
class Board {
  public:
   using piece_type = PieceType;
   using token_type = typename piece_type::token_type;
   using position_type = typename piece_type::position_type;
   using move_type = Move< position_type >;
   using map_type = std::map< position_type, sptr< PieceType > >;
   using inverse_map_type = std::unordered_map< token_type, position_type >;
   using iterator = typename map_type::iterator;
   using const_iterator = typename map_type::const_iterator;
   using inverse_iterator = typename inverse_map_type::iterator;
   using const_inverse_iterator = typename inverse_map_type::const_iterator;

   static constexpr size_t m_dim = position_type::dim;

   ///
   /// Constructors
   ///

   explicit Board(const std::array< int, m_dim > &shape);
   Board(const std::array< size_t, m_dim > &shape, const std::array< int, m_dim > &board_starts);
   Board(
      const std::array< size_t, m_dim > &shape,
      const std::vector< sptr< piece_type > > &setup_0,
      const std::vector< sptr< piece_type > > &setup_1);
   Board(
      const std::array< size_t, m_dim > &shape,
      const std::array< int, m_dim > &board_starts,
      const std::vector< sptr< piece_type > > &setup_0,
      const std::vector< sptr< piece_type > > &setup_1);
   Board(
      const std::array< size_t, m_dim > &shape,
      const std::map< position_type, token_type > &setup_0,
      const std::map< position_type, token_type > &setup_1);
   Board(
      const std::array< size_t, m_dim > &shape,
      const std::array< int, m_dim > &board_starts,
      const std::map< position_type, token_type > &setup_0,
      const std::map< position_type, token_type > &setup_1);

   virtual ~Board() = default;

   sptr< piece_type > &operator[](const position_type &position);

   const sptr< piece_type > &operator[](const position_type &position) const;

   ///
   /// Iterators
   ///

   [[nodiscard]] iterator begin() { return m_map.begin(); }

   [[nodiscard]] iterator end() { return m_map.end(); }

   [[nodiscard]] const_iterator begin() const { return m_map.begin(); }

   [[nodiscard]] const_iterator end() const { return m_map.end(); }

   [[nodiscard]] inverse_iterator begin_inverse(int team) { return m_map_inverse.at(team).begin(); }

   [[nodiscard]] inverse_iterator end_inverse(int team) { return m_map_inverse.at(team).end(); }

   [[nodiscard]] const_inverse_iterator begin_inverse(int team) const
   {
      return m_map_inverse.at(team).begin();
   }

   [[nodiscard]] const_inverse_iterator end_inverse(int team) const
   {
      return m_map_inverse.at(team).end();
   }

   ///
   /// Getters
   ///

   [[nodiscard]] auto get_shape() const { return m_shape; }

   [[nodiscard]] auto get_starts() const { return m_starts; }

   [[nodiscard]] auto size() const { return m_map.size(); }

   map_type const *get_map() const { return m_map; }

   std::array< inverse_map_type, 2 > const &get_inverse_map() const { return m_map_inverse; }

   const_inverse_iterator get_position_of_token(int team, const token_type &token) const
   {
      return m_map_inverse.at(team).find(token);
   }

   size_t get_count_of_token(int team, const token_type &token) const
   {
      return m_map_inverse.at(team).count(token);
   }

   std::vector< sptr< piece_type > > get_pieces(Team team) const;

   ///
   /// API
   ///

   std::tuple< bool, size_t > check_bounds(const position_type &pos) const;

   inline void is_within_bounds(const position_type &pos);

   void update_board(const position_type &pos, const sptr< piece_type > &pc);

   [[nodiscard]] virtual std::string print_board(Team team, bool hide_unknowns) const = 0;

   sptr< Board< piece_type > > clone() const { return sptr< Board< piece_type > >(clone_impl()); }

  protected:
   std::array< size_t, m_dim > m_shape;
   std::array< int, m_dim > m_starts;
   map_type m_map;
   std::array< inverse_map_type, 2 > m_map_inverse;

  private:
   template < class T, size_t N >
   static std::array< T, N > make_array(const T &value);

   void _fill_board_null_pieces(
      size_t dim,
      std::array< int, m_dim > &&position_pres = make_array< int, m_dim >(0));

   void _fill_inverse_board();

   virtual Board< piece_type > *clone_impl() const = 0;

   //   virtual Board< piece_type > *clone_impl() const = 0;
};

template < typename PieceType >
template < class T, size_t N >
std::array< T, N > Board< PieceType >::make_array(const T &value)
{
   // only works for default constructible value types T!
   // (intended for int in this class)
   std::array< T, N > ret;
   ret.fill(value);
   return ret;
}

template < typename PieceType >
std::tuple< bool, size_t > Board< PieceType >::check_bounds(const position_type &pos) const
{
   for(size_t i = 0; i < m_dim; ++i) {
      if(pos[i] < m_starts[i] || (pos[i] > 0 && (size_t) pos[i] >= m_shape[i])) {
         return std::make_tuple(false, i);
      }
   }
   return std::make_tuple(true, 0);
}

template < typename PieceType >
const sptr< PieceType > &Board< PieceType >::operator[](const position_type &position) const
{
   return m_map.at(position);
}

template < typename PieceType >
sptr< PieceType > &Board< PieceType >::operator[](const position_type &position)
{
   return m_map[position];
}

template < typename PieceType >
void Board< PieceType >::update_board(const position_type &pos, const sptr< piece_type > &pc_ptr)
{
   /* Note for the usage of this method:
    * If you use a board game with, in which pieces may "defeat" one another,
    * one will need to always update the piece, that did the attacking, first.
    * Second is always the defending piece. Otherwise a hard-to-trace access bug
    * in the inverse map of the kins may occur!
    */
   is_within_bounds(pos);
   auto pc_before = m_map[pos];

   if(! pc_before->is_null() && *pc_before != *pc_ptr) {
      if(int team = pc_before->get_team(); team != -1) {
         m_map_inverse[team].erase(pc_before->get_token());
      }
   }

   pc_ptr->set_position(pos);
   (*this)[pos] = pc_ptr;

   if(int team = pc_ptr->get_team(); ! pc_ptr->is_null() && team != -1) {
      m_map_inverse[team][pc_ptr->get_token()] = pos;
   }
}

template < typename PieceType >
void Board< PieceType >::_fill_board_null_pieces(
   size_t dim,
   std::array< int, m_dim > &&position_pres)
{
   if(dim > 0) {
      for(int i = m_starts[dim - 1]; i < static_cast< int >(m_starts[dim - 1] + m_shape[dim - 1]);
          ++i) {
         position_pres[dim - 1] = i;
         _fill_board_null_pieces(dim - 1, std::forward< std::array< int, m_dim > >(position_pres));
      }
   } else {
      position_type pos(position_pres);
      m_map[pos] = std::make_shared< piece_type >(pos);
   }
}

template < typename PieceType >
Board< PieceType >::Board(const std::array< int, m_dim > &shape)
    : m_shape(shape), m_starts(make_array< int, m_dim >(0)), m_map(), m_map_inverse()
{
   _fill_board_null_pieces(m_dim);
}

template < typename PieceType >
Board< PieceType >::Board(
   const std::array< size_t, m_dim > &shape,
   const std::array< int, m_dim > &board_starts)
    : m_shape(shape), m_starts(board_starts), m_map(), m_map_inverse()
{
   _fill_board_null_pieces(m_dim);
}

template < typename PieceType >
Board< PieceType >::Board(
   const std::array< size_t, m_dim > &shape,
   const std::array< int, m_dim > &board_starts,
   const std::vector< sptr< piece_type > > &setup_0,
   const std::vector< sptr< piece_type > > &setup_1)
    : Board(shape, board_starts)
{
   auto setup_unwrap = [&](const std::vector< sptr< piece_type > > &setup) {
      std::map< position_type, int > seen_pos;
      for(const auto &piece : setup) {
         position_type pos = piece->get_position();
         if(seen_pos.find(pos) != seen_pos.end()) {
            // element found
            throw std::invalid_argument(
               "Parameter setup has more than one piece for the "
               "same position (position: '"
               + pos.to_string() + "').");
         }
         seen_pos[pos] = 1;
         m_map[pos] = piece;
      }
   };
   setup_unwrap(setup_0);
   setup_unwrap(setup_1);

   _fill_inverse_board();
}

template < typename PieceType >
Board< PieceType >::Board(
   const std::array< size_t, m_dim > &shape,
   const std::vector< sptr< piece_type > > &setup_0,
   const std::vector< sptr< piece_type > > &setup_1)
    : Board(shape, decltype(m_starts){0}, setup_0, setup_1)
{
}

template < typename PieceType >
Board< PieceType >::Board(
   const std::array< size_t, m_dim > &shape,
   const std::array< int, m_dim > &board_starts,
   const std::map< position_type, token_type > &setup_0,
   const std::map< position_type, token_type > &setup_1)
    : Board(shape, board_starts)
{
   auto setup_unwrap = [&](const std::map< position_type, token_type > &setup, int team) {
      // because of the short length of the vectors they might be faster than
      // using a map (fit in cache)
      std::vector< position_type > seen_pos;
      std::vector< token_type > seen_token;
      for(auto &elem : setup) {
         position_type pos = elem.first;
         auto character = elem.second;

         if(std::find(seen_pos.begin(), seen_pos.end(), pos) != seen_pos.end()) {
            // element found
            throw std::invalid_argument(
               std::string("Parameter setup has more than one piece for the ")
               + std::string("same position (position: '") + pos.to_string() + std::string("')."));
         }
         seen_pos.emplace_back(pos);
         if(std::find(seen_token.begin(), seen_token.end(), character) != seen_token.end()) {
            throw std::invalid_argument(
               std::string("Parameter setup has more than one piece for the ")
               + std::string("same character (character: '") + character.to_string()
               + std::string("')."));
         }
         seen_token.emplace_back(character);

         m_map[pos] = std::make_shared< piece_type >(pos, character, team);
      }
   };
   setup_unwrap(setup_0, 0);
   setup_unwrap(setup_1, 1);

   _fill_inverse_board();
}

template < typename PieceType >
Board< PieceType >::Board(
   const std::array< size_t, m_dim > &shape,
   const std::map< position_type, typename piece_type::token_type > &setup_0,
   const std::map< position_type, typename piece_type::token_type > &setup_1)
    : Board(shape, make_array< int, m_dim >(0), setup_0, setup_1)
{
}

template < typename PieceType >
std::vector< sptr< typename Board< PieceType >::piece_type > > Board< PieceType >::get_pieces(
   Team team) const
{
   std::vector< sptr< piece_type > > pieces;
   for(const auto &pos_piece : m_map) {
      sptr< piece_type > piece_ptr = pos_piece.second;
      if(! piece_ptr->is_null() && piece_ptr->get_team() == team) {
         pieces.emplace_back(piece_ptr);
      }
   }
   return pieces;
}

template < class PieceType >
void Board< PieceType >::_fill_inverse_board()
{
   for(const auto &piece_ptr : m_map) {
      auto &piece = piece_ptr.second;
      if(! piece->is_null()) {
         m_map_inverse[piece->get_team()][piece->get_token()] = piece->get_position();
      }
   }
}

template < typename PieceType >
inline void Board< PieceType >::is_within_bounds(const position_type &pos)
{
   if(auto [in_bounds, idx] = check_bounds(pos); ! in_bounds) {
      std::ostringstream ss;
      std::string val_str = std::to_string(pos[idx]);
      std::string bounds_str = std::to_string(m_starts[idx]) + ", " + std::to_string(m_shape[idx]);
      ss << "Index at dimension " << std::to_string(idx) << " out of bounds "
         << "(Value :" << val_str << ", Bounds: [" << bounds_str << "])";
      throw std::invalid_argument(ss.str());
   }
}
