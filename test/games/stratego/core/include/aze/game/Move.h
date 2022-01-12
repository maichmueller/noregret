
#pragma once

#include <array>
#include <vector>

#include "Position.h"

namespace aze {

// forward declare class and operators in conjunction with number types
template < class Position >
class Move;

template < typename Number, typename Position >
Move< Position > operator/(Number n, Move< Position > pos);

template < class Position >
class Move {
  public:
   using position_type = Position;
   using move_container = std::array< Position, 2 >;
   using iterator = typename move_container::iterator;
   using const_iterator = typename move_container::const_iterator;

  private:
   move_container from_to;

  public:
   Move(const position_type& pos_from, const position_type& pos_to) : from_to{pos_from, pos_to} {}
   Move(position_type&& pos_from, position_type&& pos_to) : from_to{pos_from, pos_to} {}

   const position_type& operator[](unsigned int index) const { return from_to[index]; }
   position_type& operator[](unsigned int index) { return from_to[index]; }

   iterator begin() { return from_to.begin(); }
   const_iterator begin() const { return from_to.begin(); }
   iterator end() { return from_to.end(); }
   const_iterator end() const { return from_to.end(); }

   auto get_positions() const { return from_to; }

   Move operator+(const Move& other_move)
   {
      return Move{this[0] + other_move[0], this[1] + other_move[1]};
   }
   Move operator*(const Move& other_move)
   {
      return Move{this[0] * other_move[1], this[1] * other_move[1]};
   }
   template < typename Number >
   Move operator+(const Number& n)
   {
      return Move{this[0] * n, this[1] * n};
   }
   Move operator-(const Move& other_move) { return *this + (-1 * other_move); }
   Move operator/(const Move& other_move) { return *this * (1 / other_move); }
   bool operator==(const Move& other) const
   {
      return (*this)[0] == other[0] && (*this)[1] == other[1];
   }
   bool operator!=(const Move& other) const { return *this != other; }

   template < typename start_container, typename end_container >
   Move< position_type > invert(const start_container& starts, const end_container& ends)
   {
      Move< position_type > copy(*this);
      for(position_type& pos : copy) {
         pos = pos.invert(starts, ends);
      }
      return copy;
   };

   std::string to_string()
   {
      return from_to[0].to_string() + std::string(" -> ") + from_to[1].to_string();
   }

   friend std::ostream& operator<<(std::ostream& os, const Move< position_type > move)
   {
      os << move[0].to_string() << "->" << move[1].to_string();
      return os;
   }
};

template < typename Number, typename Position >
Move< Position > operator/(const Number& n, const Move< Position >& pos)
{
   Move< Position > m(pos);
   m[0] = 1 / m[0];
   m[1] = 1 / m[1];
   return m;
}

}  // namespace aze

namespace std {
template < typename Position >
struct hash< aze::Move< Position > > {
   constexpr size_t operator()(const aze::Move< Position >& move) const
   {
      // ( x*p1 xor y*p2 xor z*p3) mod n is supposedly a better spatial hash
      // function
      auto pos_hasher = hash< Position >();
      long int curr = pos_hasher(move[0]) * primes::primes_list[0];
      curr ^= pos_hasher(move[1]) * primes::primes_list[1];
      return curr % primes::primes_list.back();
   }
};
}  // namespace std
