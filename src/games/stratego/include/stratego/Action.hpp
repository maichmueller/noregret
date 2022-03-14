
#pragma once

#include <array>
#include <utility>
#include <vector>

#include "StrategoDefs.hpp"
#include "Utils.hpp"

namespace stratego {

class Move;

template < typename Number >
Move operator/(const Number& n, const Move& m);

class Move {
  public:
   using action_container = std::array< Position, 2 >;
   using iterator = typename action_container::iterator;
   using const_iterator = typename action_container::const_iterator;

  private:
   action_container from_to;

  public:
   Move(Position pos_from, Position pos_to) : from_to{std::move(pos_from), std::move(pos_to)} {}

   const Position& operator[](unsigned int index) const { return from_to[index]; }
   Position& operator[](unsigned int index) { return from_to[index]; }

   [[nodiscard]] auto inline from() const { return (*this)[0]; }
   [[nodiscard]] auto inline to() const { return (*this)[1]; }
   [[nodiscard]] auto inline flatten() const
   {
      return std::array{from()[0], from()[1], to()[0], to()[1]};
   }

   iterator begin() { return from_to.begin(); }
   [[nodiscard]] const_iterator begin() const { return from_to.begin(); }
   iterator end() { return from_to.end(); }
   [[nodiscard]] const_iterator end() const { return from_to.end(); }

   [[nodiscard]] auto positions() const { return from_to; }

   Move operator+(const Move& other_action) const
   {
      return {(*this)[0] + other_action[0], (*this)[1] + other_action[1]};
   }
   Move operator*(const Move& other_move) const
   {
      return {(*this)[0] * other_move[1], (*this)[1] * other_move[1]};
   }
   template < typename Number >
   Move operator+(const Number& n) const
   {
      return {(*this)[0] * n, (*this)[1] * n};
   }
   template < typename Number >
   Move operator*(const Number& n) const
   {
      return {(*this)[0] * n, (*this)[1] * n};
   }
   Move operator-(const Move& other_move) const { return *this + (other_move * (-1)); }
   Move operator/(const Move& other_move) const { return *this * (1 / other_move); }
   bool operator==(const Move& other) const
   {
      return (*this)[0] == other[0] && (*this)[1] == other[1];
   }
   bool operator!=(const Move& other) const { return not ((*this) == other); }

   [[nodiscard]] auto to_string() const { return from().to_string() + "->" + to().to_string(); }

   friend auto& operator<<(std::ostream& os, const Move& action)
   {
      os << action.from().to_string() << "->" << action.to().to_string();
      return os;
   }
   friend auto& operator<<(std::stringstream& os, const Move& action)
   {
      os << action.from().to_string() << "->" << action.to().to_string();
      return os;
   }

   template < std::size_t Index >
   std::tuple_element_t< Index, ::stratego::Move > get() const
   {
      if constexpr(Index == 0)
         return from();
      if constexpr(Index == 1)
         return to();
   }
};

template < typename Number >
Move operator/(const Number& n, const Move& m)
{
   return {n / m[0], n / m[1]};
}

class Action {
  public:
   using move_type = Move;
   using iterator = typename Move::iterator;
   using const_iterator = typename Move::const_iterator;

   Action(Team team, Move move) : m_team(team), m_move(std::move(move)) {}

   const Position& operator[](unsigned int index) const { return m_move[index]; }
   Position& operator[](unsigned int index) { return m_move[index]; }

   [[nodiscard]] inline auto team() const { return m_team; }
   [[nodiscard]] inline auto& move() const { return m_move; }
   inline auto& move() { return m_move; }

   iterator begin() { return m_move.begin(); }
   [[nodiscard]] const_iterator begin() const { return m_move.begin(); }
   iterator end() { return m_move.end(); }
   [[nodiscard]] const_iterator end() const { return m_move.end(); }

   bool operator==(const Action& other) const
   {
      return m_team == other.team() and m_move == other.move();
   }
   bool operator!=(const Action& other) const { return not ((*this) == other); }

   [[nodiscard]] auto to_string() const
   {
      return std::string(utils::enum_name(m_team)) + ":" + move().to_string();
   }

   friend auto& operator<<(std::ostream& os, const Action& action)
   {
      os << action.to_string();
      return os;
   }
   friend auto& operator<<(std::stringstream& os, const Action& action)
   {
      os << utils::enum_name(action.team()) << ":" << action.move().to_string();
      return os;
   }

   template < std::size_t Index >
   std::tuple_element_t< Index, ::stratego::Action > get() const
   {
      static_assert(Index <= 2, "An Action object decomposes into 3 parts.");

      if constexpr(Index == 0)
         return m_team;
      if constexpr(Index == 1)
         return m_move.from();
      if constexpr(Index == 2)
         return m_move.to();
   }

  private:
   Team m_team;
   Move m_move;
};

}  // namespace stratego

// allow action to be unpacked by structured bindings
namespace std {
template <>
struct tuple_size< ::stratego::Move >: integral_constant< size_t, 2 > {
};

template <>
struct tuple_element< 0, ::stratego::Move > {
   using type = ::stratego::Position;
};

template <>
struct tuple_element< 1, ::stratego::Move > {
   using type = ::stratego::Position;
};

template <>
struct tuple_size< ::stratego::Action >: integral_constant< size_t, 3 > {
};

template <>
struct tuple_element< 0, ::stratego::Action > {
   using type = ::stratego::Team;
};

template <>
struct tuple_element< 1, ::stratego::Action > {
   using type = ::stratego::Position;
};

template <>
struct tuple_element< 2, ::stratego::Action > {
   using type = ::stratego::Position;
};

template <>
struct hash< stratego::Move > {
   size_t operator()(const stratego::Move& move) const
   {
      std::stringstream ss;
      for(const auto& pos : move) {
         for(const auto& value : pos) {
            ss << value << ",";
         }
      }
      return std::hash< std::string >{}(ss.str());
   }
};

template <>
struct hash< stratego::Action > {
   size_t operator()(const stratego::Action& action) const
   {
      std::stringstream ss;
      ss << action.team() << "\n";
      for(const auto& pos : action.move()) {
         for(const auto& value : pos) {
            ss << value << ",";
         }
      }
      return std::hash< std::string >{}(ss.str());
   }
};

}  // namespace std
