

#pragma once

#include <array>
#include <sstream>
#include <string>
#include <vector>

#include "aze/utils/prime_list.h"


namespace aze {

// forward declare class and operators in conjunction with number types
template < typename ValueType, size_t N >
class Position;

template < typename Number, typename ValueType, size_t N >
Position< ValueType, N > operator*(const Number &n, const Position< ValueType, N > &pos);

template < typename Number, typename ValueType, size_t N >
Position< ValueType, N > operator/(const Number &n, const Position< ValueType, N > &pos);

// actual class definition
template < typename ValueType, size_t N >
class Position {
  public:
   using value_type = ValueType;
   using container_type = std::array< value_type, N >;
   using Iterator = typename container_type::iterator;
   using ConstIterator = typename container_type::const_iterator;
   static constexpr size_t dim = N;

  private:
   container_type m_coordinates;

   template < size_t... Indices, typename... Types >
   Position(std::index_sequence< Indices... >, Types &&...args)
   {
      // c++17 fold expression
      (static_cast< void >(m_coordinates[Indices] = args), ...);
   }

  public:
   /*
    * Constructor that allows to initialize the position by typing out all N
    * coordinates. For details on how this implementation works read up on
    * 'SFINAE'.
    */
   template < typename... Types, typename std::enable_if< sizeof...(Types) == N, int >::type = 0 >
   Position(Types &&...args)
       : Position(std::index_sequence_for< Types... >{}, std::forward< Types >(args)...)
   {
   }

   Position() : m_coordinates() {}

   Position(const Position &position) : m_coordinates(position.get_coordinates()) {}

   explicit Position(container_type coords) : m_coordinates(std::move(coords)) {}

   const value_type &operator[](unsigned int index) const { return m_coordinates[index]; }

   value_type &operator[](unsigned int index) { return m_coordinates[index]; }

   Iterator begin() { return m_coordinates.begin(); }

   ConstIterator begin() const { return m_coordinates.begin(); }

   Iterator end() { return m_coordinates.end(); }

   ConstIterator end() const { return m_coordinates.end(); }

   Position< value_type, N > operator+(const Position< value_type, N > &pos) const;

   Position< value_type, N > operator-(const Position< value_type, N > &pos) const;

   Position< value_type, N > operator*(const Position< value_type, N > &pos) const;

   Position< value_type, N > operator/(const Position< value_type, N > &pos) const;

   template < typename Number >
   Position< value_type, N > operator+(const Number &n) const;

   template < typename Number >
   Position< value_type, N > operator-(const Number &n) const;

   template < typename Number >
   Position< value_type, N > operator*(const Number &n) const;

   template < typename Number >
   Position< value_type, N > operator/(const Number &n) const;

   bool operator==(const Position &other) const;

   bool operator!=(const Position &other) const;

   bool operator<(const Position &other) const;

   bool operator<=(const Position &other) const;

   bool operator>(const Position &other) const;

   bool operator>=(const Position &other) const;

   container_type get_coordinates() const { return m_coordinates; }

   template < typename container_start, typename container_end >
   Position invert(const container_start &starts, const container_end &ends);

   [[nodiscard]] std::string to_string() const;
};

// free operators for switched call positions

template < typename Number, typename ValueType, size_t N >
Position< ValueType, N > operator*(const Number &n, const Position< ValueType, N > &pos)
{
   return pos * n;
}

template < typename Number, typename ValueType, size_t N >
Position< ValueType, N > operator/(const Number &n, const Position< ValueType, N > &pos)
{
   Position< ValueType, N > p(pos);
   for(size_t i = 0; i < N; ++i) {
      p[i] = n / p[i];
   }
   return p;
}

// method implementations

template < typename ValueType, size_t N >
Position< ValueType, N > Position< ValueType, N >::operator+(
   const Position< ValueType, N > &pos) const
{
   Position< ValueType, N > p(*this);
   for(size_t i = 0; i < N; ++i) {
      p[i] += pos[i];
   }
   return p;
}

template < typename ValueType, size_t N >
Position< ValueType, N > Position< ValueType, N >::operator-(
   const Position< ValueType, N > &pos) const
{
   return *this + (-1 * pos);
}

template < typename ValueType, size_t N >
Position< ValueType, N > Position< ValueType, N >::operator*(
   const Position< ValueType, N > &pos) const
{
   Position< ValueType, N > p(*this);
   for(size_t i = 0; i < m_coordinates.size(); ++i) {
      p[i] *= pos[i];
   }
   return p;
}

template < typename ValueType, size_t N >
Position< ValueType, N > Position< ValueType, N >::operator/(
   const Position< ValueType, N > &pos) const
{
   Position< ValueType, N > p(*this);
   for(size_t i = 0; i < N; ++i) {
      p[i] /= pos[i];
   }
   return p;
}

template < typename ValueType, size_t N >
template < typename Number >
Position< ValueType, N > Position< ValueType, N >::operator+(const Number &n) const
{
   Position< ValueType, N > p(*this);
   for(size_t i = 0; i < N; ++i) {
      p[i] += n;
   }
   return p;
}

template < typename ValueType, size_t N >
template < typename Number >
Position< ValueType, N > Position< ValueType, N >::operator-(const Number &n) const
{
   return *this + (-n);
}

template < typename ValueType, size_t N >
template < typename Number >
Position< ValueType, N > Position< ValueType, N >::operator*(const Number &n) const
{
   Position< ValueType, N > p(*this);
   for(size_t i = 0; i < N; ++i) {
      p[i] *= n;
   }
   return p;
}

template < typename ValueType, size_t N >
template < typename Number >
Position< ValueType, N > Position< ValueType, N >::operator/(const Number &n) const
{
   return (*this) * (1 / n);
}

template < typename ValueType, size_t N >
bool Position< ValueType, N >::operator==(const Position &other) const
{
   for(size_t i = 0; i < N; ++i) {
      if((*this)[i] != other[i])
         return false;
   }
   return true;
}

template < typename ValueType, size_t N >
bool Position< ValueType, N >::operator!=(const Position &other) const
{
   return *this != other;
}

template < typename ValueType, size_t N >
bool Position< ValueType, N >::operator<(const Position &other) const
{
   for(size_t i = 0; i < N; ++i) {
      if((*this)[i] < other[i])
         return true;
      else if((*this)[i] > other[i])
         return false;
      // the else case is the == case for which we simply continue onto the next
      // dimension
   }
   return false;
}

template < typename ValueType, size_t N >
bool Position< ValueType, N >::operator<=(const Position &other) const
{
   for(size_t i = 0; i < N; ++i) {
      if((*this)[i] == other[i])
         continue;
      else
         return (*this)[i] <= other[i];
   }
   return true;
}

template < typename ValueType, size_t N >
bool Position< ValueType, N >::operator>(const Position &other) const
{
   for(size_t i = 0; i < N; ++i) {
      if((*this)[i] > other[i])
         return true;
      else if((*this)[i] < other[i])
         return false;
      // the else case is the == case for which we simply continue onto the next
      // dimension
   }
   return false;
}

template < typename ValueType, size_t N >
bool Position< ValueType, N >::operator>=(const Position &other) const
{
   for(size_t i = 0; i < N; ++i) {
      if((*this)[i] == other[i])
         continue;
      else
         return (*this)[i] >= other[i];
   }
   return true;
}

template < typename ValueType, size_t N >
std::string Position< ValueType, N >::to_string() const
{
   std::stringstream ss;
   ss << "(";
   for(size_t i = 0; i < N - 1; ++i) {
      ss << std::to_string(m_coordinates[i]) << ", ";
   }
   ss << std::to_string(m_coordinates.back()) << ")";
   return ss.str();
}

template < typename ValueType, size_t N >
template < typename container_start, typename container_end >
Position< ValueType, N > Position< ValueType, N >::invert(
   const container_start &starts,
   const container_end &ends)
{
   if constexpr(std::is_floating_point_v< ValueType >) {
      if constexpr(! std::is_floating_point_v< typename container_start::value_type >) {
         throw std::invalid_argument(
            std::string("Container value_type of 'starts' is not of floating point (")
            + std::string(typeid(typename container_start::value_type).name())
            + std::string("), while 'Position' value type is (")
            + std::string(typeid(ValueType).name()) + std::string(")."));
      }
      if constexpr(! std::is_floating_point_v< typename container_end::value_type >) {
         throw std::invalid_argument(
            std::string("Container value_type of 'ends' is not of floating point (")
            + std::string(typeid(typename container_end::value_type).name())
            + std::string("), while 'Position' value type is (")
            + std::string(typeid(ValueType).name()) + std::string(")."));
      }
   }

   Position< ValueType, N > inverted;
   for(size_t i = 0; i < N; ++i) {
      inverted[i] = starts[i] + (ends[i] - 1) - m_coordinates[i];
   }
   return inverted;
}


}

namespace std {
template < typename ValueType, size_t N >
struct hash< aze::Position< ValueType, N > > {
   constexpr size_t operator()(const aze::Position< ValueType, N > &pos) const
   {
      // ( x*p1 xor y*p2 xor z*p3) mod n is supposedly a better spatial hash
      // function
      long int curr = pos[0] * primes::primes_list[0];
      for(size_t i = 1; i < N; ++i) {
         curr ^= pos[i] * primes::primes_list[i];
      }
      return curr % primes::primes_list.back();
   }
};
}  // namespace std
