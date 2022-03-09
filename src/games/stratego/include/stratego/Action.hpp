
#pragma once

#include <array>
#include <utility>
#include <vector>

#include "StrategoDefs.hpp"

namespace stratego {

class Action;

template < typename Number >
Action operator/(const Number& n, const Action& m);

class Action {
  public:
   using action_container = std::array< Position, 2 >;
   using iterator = typename action_container::iterator;
   using const_iterator = typename action_container::const_iterator;

  private:
   action_container from_to;
   Team m_team;

  public:
   Action(const Position& pos_from, const Position& pos_to) : from_to{pos_from, pos_to} {}

   const Position& operator[](unsigned int index) const { return from_to[index]; }
   Position& operator[](unsigned int index) { return from_to[index]; }

   [[nodiscard]] auto inline team() const { return m_team; }

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

   Action operator+(const Action& other_action) const
   {
      return {(*this)[0] + other_action[0], (*this)[1] + other_action[1]};
   }
   Action operator*(const Action& other_action) const
   {
      return {(*this)[0] * other_action[1], (*this)[1] * other_action[1]};
   }
   template < typename Number >
   Action operator+(const Number& n) const
   {
      return {(*this)[0] * n, (*this)[1] * n};
   }
   template < typename Number >
   Action operator*(const Number& n) const
   {
      return {(*this)[0] * n, (*this)[1] * n};
   }
   Action operator-(const Action& other_action) const { return *this + (other_action * (-1)); }
   Action operator/(const Action& other_action) const { return *this * (1 / other_action); }
   bool operator==(const Action& other) const
   {
      return (*this)[0] == other[0] && (*this)[1] == other[1];
   }
   bool operator!=(const Action& other) const { return not ((*this) == other); }

   friend auto& operator<<(std::ostream& os, const Action& action)
   {
      os << action.from().to_string() << "->" << action.to().to_string();
      return os;
   }
   friend auto& operator<<(std::stringstream& os, const Action& action)
   {
      os << action.from().to_string() << "->" << action.to().to_string();
      return os;
   }

   template < std::size_t Index >
   std::tuple_element_t< Index, ::stratego::Action > get() const
   {
      if constexpr(Index == 0)
         return from();
      if constexpr(Index == 1)
         return to();
   }
};

template < typename Number >
Action operator/(const Number& n, const Action& m)
{
   return {n / m[0], n / m[1]};
}

}  // namespace stratego

// allow action to be unpacked by structured bindings
namespace std {
template <>
struct tuple_size< ::stratego::Action >: integral_constant< size_t, 2 > {
};

template <>
struct tuple_element< 0, ::stratego::Action > {
   using type = ::stratego::Position;
};

template <>
struct tuple_element< 1, ::stratego::Action > {
   using type = ::stratego::Position;
};

template <>
struct hash< stratego::Action > {
   size_t operator()(const stratego::Action& action) const
   {
      std::stringstream ss;
      for(const auto& pos : action) {
         for(const auto& value : pos) {
            ss << value << ",";
         }
      }
      return std::hash<std::string>{}(ss.str());
   }
};

}  // namespace std
