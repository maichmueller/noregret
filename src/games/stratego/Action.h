
#pragma once

#include <array>
#include <utility>
#include <vector>
#include <xtensor/xio.hpp>

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

  public:
   Action(const Position& pos_from, const Position& pos_to) : from_to{pos_from, pos_to} {}

   const Position& operator[](unsigned int index) const { return from_to[index]; }
   Position& operator[](unsigned int index) { return from_to[index]; }

   [[nodiscard]] auto from() const {return (*this)[0];}
   [[nodiscard]] auto to() const {return (*this)[1];}

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
};

template < typename Number >
Action operator/(const Number& n, const Action& m)
{
   return {1 / m[0], 1 / m[1]};
}

}  // namespace stratego
